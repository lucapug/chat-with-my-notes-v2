"""LLM-as-judge for RAG evaluation.

Migrated from master_run_eval.py (sprint1-hermes-backup).
HTTP via httpx (async). Uses settings for LLM endpoints.
"""

import json
import logging
import re

import httpx

from config import settings

logger = logging.getLogger(__name__)

JUDGE_RUBRIC_WITH_GT = (
    "Sei un giudice RAG. Confronta la risposta generata dal sistema RAG "
    "con la risposta attesa (ground truth) fornita dal proprietario del vault.\n\n"
    "La risposta attesa è la verità. Se il sistema RAG dice 'non presente' "
    "ma la risposta attesa contiene informazioni concrete, il sistema ha FALLITO "
    "il retrieval, non ha avuto successo.\n\n"
    "IMPORTANTE: Valuta la risposta rispetto allo SCOPE DELLA DOMANDA, non rispetto alla completezza della ground truth. "
    "Se la domanda chiede un dato specifico e la risposta lo fornisce correttamente, completeness deve essere 5, "
    "anche se la ground truth contiene dettagli aggiuntivi non richiesti dalla domanda.\n\n"
    "Assegna un punteggio da 1 a 5 per ogni criterio:\n"
    "- accuracy: la risposta RAG è fattualmente equivalente alla risposta attesa rispetto alla domanda? "
    "(1=completamente diversa, 5=equivalente)\n"
    "- completeness: la risposta RAG copre tutte le informazioni richieste dalla domanda? "
    "(1=mancano informazioni critiche, 5=tutte le informazioni richieste presenti)\n"
    "- hallucination: la risposta RAG contiene informazioni false o contraddittorie? "
    "(1=estese invenzioni o falsità, 5=nessuna allucinazione). NOTA: Non penalizzare informazioni extra vere che non sono nella ground truth.\n"
    "- relevance: la risposta RAG è pertinente alla domanda? "
    "(1=fuori tema, 5=completamente pertinente)\n\n"
    "Outcome logic:\n"
    "- fail: accuracy <= 2 OR hallucination <= 2\n"
    "- warning: tutti >= 3 ma almeno uno < 5, oppure completeness <= 2 con accuracy >= 4\n"
    "- pass: accuracy >= 4 AND completeness >= 4 AND hallucination >= 4 AND relevance >= 4\n\n"
    "Rispondi SOLO con JSON in questo formato, nessun altro testo:\n"
    '{"accuracy": N, "completeness": N, "hallucination": N, '
    '"relevance": N, "outcome": "pass|warning|fail", '
    '"reasoning": "breve motivazione in italiano"}'
)

JUDGE_RUBRIC_NO_GT = (
    "Valuta la risposta rispetto alla domanda e al contesto fornito.\n"
    "Assegna un punteggio da 1 a 5 per ogni criterio:\n"
    "- accuracy: la risposta è fattualmente corretta rispetto al contesto?\n"
    "- completeness: copre tutti gli aspetti rilevanti della domanda?\n"
    "- hallucination: la risposta inventa fatti non presenti nel contesto? "
    "(5=nessuna allucinazione)\n"
    "- relevance: la risposta è pertinente alla domanda?\n\n"
    "Outcome logic:\n"
    "- fail: se qualsiasi criterio <= 2\n"
    "- warning: tutti >= 3 ma almeno uno < 5\n"
    "- pass: tutti = 5\n\n"
    "Rispondi SOLO con JSON in questo formato, nessun altro testo:\n"
    '{"accuracy": N, "completeness": N, "hallucination": N, '
    '"relevance": N, "outcome": "pass|warning|fail", '
    '"reasoning": "breve motivazione"}'
)


def _normalize_text_field(value: str | list | dict | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        normalized_parts: list[str] = []
        for item in value:
            normalized_parts.append(_normalize_text_field(item))
        return "\n".join([p for p in normalized_parts if p])
    if isinstance(value, dict):
        if "text" in value or "content" in value:
            return _normalize_text_field(value.get("text") or value.get("content"))
        normalized_parts: list[str] = []
        for item in value.values():
            normalized_parts.append(_normalize_text_field(item))
        return "\n".join([p for p in normalized_parts if p])
    return str(value)


def _try_load_json(candidate: str) -> dict | None:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None


def _extract_json_with_outcome(text: str) -> dict | None:
    if not text:
        return None

    cleaned = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*", "", cleaned, flags=re.IGNORECASE)

    match = re.search(r"\{.*?\"outcome\".*?\}", cleaned, re.DOTALL)
    if match:
        parsed = _try_load_json(match.group(0))
        if parsed is not None:
            return parsed

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        parsed = _try_load_json(fenced.group(1))
        if parsed is not None:
            return parsed

    # Partial JSON / truncated output fallback: extract numeric values from available text.
    partial = _extract_scores_from_text(cleaned)
    if partial is not None:
        return partial

    # Balanced braces fallback.
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for i, char in enumerate(cleaned[start:], start=start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    if "\"outcome\"" in candidate:
                        parsed = _try_load_json(candidate)
                        if parsed is not None:
                            return parsed
                    break
        start = cleaned.find("{", start + 1)

    return None


def _extract_scores_from_text(text: str) -> dict | None:
    scores: dict[str, int | str] = {}
    for criterion in ["accuracy", "completeness", "hallucination", "relevance"]:
        m = re.search(rf'"?{criterion}"?\s*[:=]\s*(\d+)', text, re.IGNORECASE)
        if m:
            scores[criterion] = int(m.group(1))

    if not scores:
        return None

    outcome_match = re.search(r'"?outcome"?\s*[:=]\s*"(pass|warning|fail)"', text, re.IGNORECASE)
    if outcome_match:
        outcome = outcome_match.group(1).lower()
    elif re.search(r"\bpass\b", text, re.IGNORECASE):
        outcome = "pass"
    elif re.search(r"\bfail\b", text, re.IGNORECASE):
        outcome = "fail"
    else:
        outcome = "warning"

    reasoning_match = re.search(r'"?reasoning"?\s*[:=]\s*"([^"]*)"', text, re.IGNORECASE)
    if reasoning_match:
        scores["reasoning"] = reasoning_match.group(1)

    return {**scores, "outcome": outcome}


def extract_judge_scores(msg: dict) -> dict:
    """Extract structured scores from judge response.

    Handles non-ideal judge output, including JSON blocks, markdown fences,
    and message content encoded as lists or nested objects.
    """
    content = _normalize_text_field(msg.get("content"))
    reasoning = _normalize_text_field(msg.get("reasoning"))

    for text in [content, reasoning]:
        parsed = _extract_json_with_outcome(text)
        if parsed is not None:
            return parsed

    partial = _extract_scores_from_text(f"{content}\n{reasoning}")
    if partial is not None:
        return partial

    # Regex fallback: extract individual numeric scores from reasoning or content.
    scores: dict = {}
    for criterion in ["accuracy", "completeness", "hallucination", "relevance"]:
        for source in [reasoning, content]:
            m = re.search(
                rf"\b{criterion}\b\s*[:=]\s*(\d+)", source, re.IGNORECASE
            )
            if m:
                scores[criterion] = int(m.group(1))
                break

    if scores:
        combined = f"{reasoning}\n{content}".lower()
        if re.search(r"\bpass\b", combined):
            outcome = "pass"
        elif re.search(r"\bfail\b", combined):
            outcome = "fail"
        else:
            outcome = "warning"
        return {**scores, "outcome": outcome, "note": "extracted from reasoning/content"}

    return {
        "parse_error": True,
        "raw_content": content[:300],
        "raw_reasoning": reasoning[:300],
    }


async def judge_answer(
    query: str,
    answer: str,
    chunks: list[dict],
    ground_truth: str | None = None,
) -> dict:
    """Judge a RAG answer using the Ollama judge model.

    Uses JUDGE_RUBRIC_WITH_GT when ground_truth is provided,
    JUDGE_RUBRIC_NO_GT otherwise.

    Returns a dict with keys: accuracy, completeness, hallucination,
    relevance, outcome, reasoning (or parse_error on extraction failure).
    """
    rubric = JUDGE_RUBRIC_WITH_GT if ground_truth else JUDGE_RUBRIC_NO_GT

    if ground_truth:
        user_msg = (
            f"Domanda: {query}\n\n"
            f"Risposta attesa (ground truth):\n{ground_truth}\n\n"
            f"Risposta del sistema RAG:\n{answer}"
        )
    else:
        context_parts = [
            f"--- CHUNK {i + 1} ---\n"
            f"Titolo: {c.get('title', '')}\n"
            f"Fonte: {c.get('file', '')}\n"
            f"Testo: {c.get('text', '')[:2000]}"
            for i, c in enumerate(chunks[:5])
        ]
        user_msg = (
            f"Domanda: {query}\n\n"
            f"Contesto:\n{''.join(context_parts)}\n\n"
            f"Risposta del sistema RAG:\n{answer}"
        )

    payload = {
        "model": settings.ollama_judge_model,
        "messages": [
            {"role": "system", "content": rubric},
            {"role": "user", "content": user_msg},
        ],
        "num_predict": 4096,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(
                f"{settings.ollama_judge_url}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            logger.error(
                "judge_answer: timeout after 300 seconds, model=%s, query=%s",
                settings.ollama_judge_model,
                query,
            )
            raise
        except httpx.HTTPStatusError as exc:
            logger.error(
                "judge_answer: HTTP error status=%s body=%s",
                exc.response.status_code,
                exc.response.text,
            )
            raise
        except ValueError as exc:
            logger.error(
                "judge_answer: invalid JSON response body=%s",
                resp.text if 'resp' in locals() else '<no response>',
            )
            raise

    return extract_judge_scores(data["choices"][0]["message"])
