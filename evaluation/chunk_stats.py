"""Analyze chunk length distribution for the current vault index.

This script loads the existing BRF vault index from `vault_index.pkl` and
reports the token-length distribution of indexed chunks.

When to run:
- before re-ingest to compare the new chunking strategy against the
  current distribution
- monthly to detect drift in chunk size distribution
- any time you want a quick health check of the vault index

Run from the project root:
    python evaluation/chunk_stats.py
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import minsearch
from dotenv import load_dotenv


DEFAULT_VAULT_INDEX_PATH = Path("data/vault_index.pkl")
ENV_VAR_NAME = "VAULT_INDEX_PATH"
HISTOGRAM_WIDTH = 40
THRESHOLDS = [200, 400, 600, 800]


@dataclass(frozen=True)
class ChunkStats:
    total: int
    minimum: int
    maximum: int
    mean: float
    median: float
    percentiles: dict[int, int]
    threshold_counts: dict[int, int]
    threshold_percents: dict[int, float]
    histogram_bins: list[tuple[int, int, int]]


def load_index_path() -> Path:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    raw = os.getenv(ENV_VAR_NAME)
    if raw:
        path = Path(raw)
    else:
        path = DEFAULT_VAULT_INDEX_PATH

    if not path.is_absolute():
        path = Path.cwd() / path

    return path


def load_chunks(index_path: Path) -> list[dict[str, object]]:
    index = minsearch.AppendableIndex.load(str(index_path))
    docs = getattr(index, "docs", None)
    if docs is None:
        raise RuntimeError("Loaded index does not expose `docs`.")
    return docs


def count_tokens(text: str) -> int:
    return len(text.split())


def percentile(values: list[int], percent: float) -> int:
    if not values:
        return 0
    if percent <= 0:
        return values[0]
    if percent >= 100:
        return values[-1]
    rank = percent / 100 * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    fraction = rank - lower
    if lower == upper:
        return values[int(rank)]
    return round(values[lower] + (values[upper] - values[lower]) * fraction)


def build_histogram(values: list[int], bins: int = 10) -> list[tuple[int, int, int]]:
    if not values:
        return []

    low = min(values)
    high = max(values)
    if low == high:
        return [(low, high, len(values))]

    span = high - low + 1
    step = math.ceil(span / bins)
    bucket_counts = [0] * bins

    for value in values:
        index = min((value - low) // step, bins - 1)
        bucket_counts[index] += 1

    histogram = []
    for i, count in enumerate(bucket_counts):
        start = low + i * step
        end = min(low + (i + 1) * step - 1, high)
        histogram.append((start, end, count))

    return histogram


def calculate_stats(token_counts: list[int]) -> ChunkStats:
    sorted_counts = sorted(token_counts)
    total = len(sorted_counts)
    percentiles_map = {
        25: percentile(sorted_counts, 25),
        50: percentile(sorted_counts, 50),
        75: percentile(sorted_counts, 75),
        90: percentile(sorted_counts, 90),
        95: percentile(sorted_counts, 95),
        99: percentile(sorted_counts, 99),
    }
    threshold_counts = {
        threshold: sum(1 for value in sorted_counts if value > threshold)
        for threshold in THRESHOLDS
    }
    threshold_percents = {
        threshold: (count / total * 100 if total else 0.0)
        for threshold, count in threshold_counts.items()
    }
    histogram_bins = build_histogram(sorted_counts)

    return ChunkStats(
        total=total,
        minimum=sorted_counts[0],
        maximum=sorted_counts[-1],
        mean=mean(sorted_counts) if sorted_counts else 0.0,
        median=median(sorted_counts) if sorted_counts else 0.0,
        percentiles=percentiles_map,
        threshold_counts=threshold_counts,
        threshold_percents=threshold_percents,
        histogram_bins=histogram_bins,
    )


def render_histogram(histogram: list[tuple[int, int, int]]) -> str:
    if not histogram:
        return "(no data)"
    max_count = max(count for _, _, count in histogram)
    lines = []
    for start, end, count in histogram:
        bar_length = round((count / max_count) * HISTOGRAM_WIDTH) if max_count else 0
        bar = "#" * bar_length
        lines.append(f"{start:6d} - {end:6d} | {count:5d} | {bar}")
    return "\n".join(lines)


def render_report(path: Path, stats: ChunkStats) -> str:
    lines = [
        f"Vault index path: {path}",
        f"Total chunks: {stats.total}",
        f"Minimum tokens: {stats.minimum}",
        f"Maximum tokens: {stats.maximum}",
        f"Mean tokens: {stats.mean:.2f}",
        f"Median tokens: {stats.median:.2f}",
        "",
        "Percentiles:",
    ]

    for pct, value in stats.percentiles.items():
        lines.append(f"  {pct:2d}th: {value}")

    lines.extend([
        "",
        "Thresholds:",
        "Threshold | Count | Percent",
        "----------|-------|--------",
    ])
    for threshold in THRESHOLDS:
        lines.append(
            f"    >{threshold:3d} | {stats.threshold_counts[threshold]:5d} | {stats.threshold_percents[threshold]:6.2f}%"
        )

    lines.extend([
        "",
        "ASCII histogram:",
        render_histogram(stats.histogram_bins),
    ])

    return "\n".join(lines)


def main() -> int:
    index_path = load_index_path()
    if not index_path.exists():
        raise FileNotFoundError(f"Vault index not found at {index_path}")

    chunks = load_chunks(index_path)
    token_counts = [count_tokens(str(chunk.get("text", ""))) for chunk in chunks]
    stats = calculate_stats(token_counts)
    print(render_report(index_path, stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
