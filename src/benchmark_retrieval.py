from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from common import project_path
from retriever import HybridRetriever


def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("El benchmark debe ser una lista JSON.")
    return data


def evaluate_mode(retriever: HybridRetriever, queries: list[dict[str, Any]], mode: str, top_k: int) -> tuple[list[dict], dict]:
    rows: list[dict[str, Any]] = []
    reciprocal_ranks: list[float] = []
    hits_at_1: list[int] = []
    hits_at_3: list[int] = []
    hits_at_5: list[int] = []

    for item in queries:
        expected = {int(x) for x in item["expected_cantos"]}
        hits = retriever.search(item["query"], top_k=top_k, mode=mode)
        first_rank = next((rank for rank, hit in enumerate(hits, start=1) if hit.canto in expected), None)
        rr = 0.0 if first_rank is None else 1.0 / first_rank
        reciprocal_ranks.append(rr)
        hits_at_1.append(int(first_rank is not None and first_rank <= 1))
        hits_at_3.append(int(first_rank is not None and first_rank <= 3))
        hits_at_5.append(int(first_rank is not None and first_rank <= 5))
        rows.append({
            "query_id": item["query_id"],
            "mode": mode,
            "query": item["query"],
            "expected_cantos": ",".join(str(x) for x in sorted(expected)),
            "first_relevant_rank": first_rank or "",
            "reciprocal_rank": rr,
            "top_chunk_ids": "|".join(hit.chunk_id for hit in hits),
            "top_cantos": "|".join(str(hit.canto) for hit in hits),
            "top_scores": "|".join(f"{hit.score:.8f}" for hit in hits),
        })

    metrics = {
        "mode": mode,
        "queries": len(queries),
        "hit_at_1": mean(hits_at_1),
        "hit_at_3": mean(hits_at_3),
        "hit_at_5": mean(hits_at_5),
        "mrr": mean(reciprocal_ranks),
    }
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa recuperación densa, léxica e híbrida.")
    parser.add_argument("--queries", default="data/benchmark/retrieval_queries.json")
    parser.add_argument("--config", default="config/retrieval.yaml")
    parser.add_argument("--sqlite", default="data/processed/corpus_odisea.sqlite")
    parser.add_argument("--qdrant-mode", choices=["server", "local", "memory"], default="local")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--local-path", default="qdrant_storage")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--embedding-backend", choices=["fastembed", "hashing"], default="fastembed")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    queries = load_queries(project_path(args.queries))
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever(
        config_path=project_path(args.config),
        sqlite_path=project_path(args.sqlite),
        mode=args.qdrant_mode,
        qdrant_url=args.qdrant_url,
        local_path=args.local_path,
        embedding_backend=args.embedding_backend,
    )
    all_rows: list[dict] = []
    all_metrics: list[dict] = []
    try:
        for mode in ("dense", "lexical", "hybrid"):
            rows, metrics = evaluate_mode(retriever, queries, mode, args.top_k)
            all_rows.extend(rows)
            all_metrics.append(metrics)
            print(metrics)
    finally:
        retriever.close()

    details_path = output_dir / "retrieval_results.csv"
    with details_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    metrics_path = output_dir / "retrieval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, ensure_ascii=False, indent=2)
    print(f"Resultados: {details_path}")
    print(f"Métricas: {metrics_path}")


if __name__ == "__main__":
    main()
