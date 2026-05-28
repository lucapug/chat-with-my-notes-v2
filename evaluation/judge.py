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
    "Assegna un punteggio da 1 a 5 per ogni criterio:\n"
    "- accuracy: la risposta RAG è fattualmente equivalente alla risposta attesa? "
    "(1=completamente diversa, 5=equivalente)\n"
    "- completeness: la risposta RAG copre tutti i dettagli presenti nella "
    "risposta attesa? (1=vuota o mancante, 5=tutti i dettagli presenti)\n"
    "- hallucination: la risposta RAG contiene informazioni NON presenti nella "
    "risposta attesa? (1=estese invenzioni, 5=nessuna invenzione)\n"
    "- relevance: la risposta RAG è pertinente alla domanda? "
    "(1=fuori tema, 5=completamente pertinente)\n\n"
    "Outcome logic:\n"
    "- fail: se qualsiasi criterio <= 2\n"
    "- warning: tutti >= 3 ma almeno uno < 5\n"
    "- pass: tutti = 5\n\n"
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


def extract_judge_scores(msg: dict) -> dict:
    """Extract structured scores from judge response.

    Handles gemma4:e4b thinking mode (content may be empty; falls back
    to reasoning). Strategy:
      1. Find JSON with 'outcome' key in content
      2. Find JSON in reasoning
      3. Regex-extract individual scores from reasoning text
      4. Return parse_error flag as last resort
    """
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""

    for text in [content, reasoning]:
        cleaned = re.sub(r"```json\s*", "", text)
        cleaned = re.sub(r"```\s*", "", cleaned)
        match = re.search(r'\{.*?"outcome".*?\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    # Regex fallback: extract individual numeric scores from reasoning
    scores: dict = {}
    for criterion in ["accuracy", "completeness", "hallucination", "relevance"]:
        m = re.search(
            rf"\**{criterion}\**:\s*\**\s*(\d+)", reasoning, re.IGNORECASE
        )
        if m:
            scores[criterion] = int(m.group(1))

    if scores:
        tail = reasoning.lower()[-200:]
        outcome = "warning"
        if "pass" in tail:
            outcome = "pass"
        elif "fail" in tail:
            outcome = "fail"
        return {**scores, "outcome": outcome, "note": "extracted from reasoning"}

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
        "max_tokens": 2048,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{settings.ollama_judge_url}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    return extract_judge_scores(data["choices"][0]["message"])
