from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from common import get_env, project_path
from hierarchical_retriever import HierarchicalRetriever


def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("El benchmark debe ser una lista JSON.")
    return data


def evaluate(retriever: HierarchicalRetriever, queries: list[dict[str, Any]], top_k: int) -> tuple[list[dict], dict, dict]:
    rows: list[dict[str, Any]] = []
    reciprocal_ranks: list[float] = []
    hits_at_1: list[int] = []
    hits_at_3: list[int] = []
    hits_at_5: list[int] = []
    canto_hits_at_1: list[int] = []
    canto_hits_at_3: list[int] = []

    for item in queries:
        expected = {int(x) for x in item["expected_cantos"]}
        canto_hits = retriever.select_cantos(item["query"])
        selected_cantos = [h.canto for h in canto_hits]
        first_canto_rank = next((rank for rank, canto in enumerate(selected_cantos, start=1) if canto in expected), None)
        canto_hits_at_1.append(int(first_canto_rank is not None and first_canto_rank <= 1))
        canto_hits_at_3.append(int(first_canto_rank is not None and first_canto_rank <= 3))

        hits = retriever.search(item["query"], top_k=top_k)
        first_rank = next((rank for rank, hit in enumerate(hits, start=1) if hit.canto in expected), None)
        rr = 0.0 if first_rank is None else 1.0 / first_rank
        reciprocal_ranks.append(rr)
        hits_at_1.append(int(first_rank is not None and first_rank <= 1))
        hits_at_3.append(int(first_rank is not None and first_rank <= 3))
        hits_at_5.append(int(first_rank is not None and first_rank <= 5))
        rows.append({
            "query_id": item["query_id"],
            "mode": "hierarchical",
            "query": item["query"],
            "expected_cantos": ",".join(str(x) for x in sorted(expected)),
            "selected_cantos": "|".join(str(x) for x in selected_cantos),
            "first_selected_canto_rank": first_canto_rank or "",
            "first_relevant_rank": first_rank or "",
            "reciprocal_rank": rr,
            "top_chunk_ids": "|".join(hit.chunk_id for hit in hits),
            "top_cantos": "|".join(str(hit.canto) for hit in hits),
            "top_scores": "|".join(f"{hit.score:.8f}" for hit in hits),
            "top_sources": "|".join(hit.source for hit in hits),
        })

    chunk_metrics = {
        "mode": "hierarchical",
        "queries": len(queries),
        "hit_at_1": mean(hits_at_1),
        "hit_at_3": mean(hits_at_3),
        "hit_at_5": mean(hits_at_5),
        "mrr": mean(reciprocal_ranks),
    }
    canto_metrics = {
        "mode": "canto_selection",
        "queries": len(queries),
        "hit_at_1": mean(canto_hits_at_1),
        "hit_at_3": mean(canto_hits_at_3),
    }
    return rows, chunk_metrics, canto_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el recuperador jerárquico de La Odisea.")
    parser.add_argument("--queries", default="data/benchmark/retrieval_queries_test.json")
    parser.add_argument("--config", default="config/retrieval_hierarchical.yaml")
    parser.add_argument("--sqlite", default=get_env("SQLITE_PATH", "data/processed/corpus_odisea.sqlite"))
    parser.add_argument("--profiles", default=get_env("CANTO_PROFILES_PATH", "data/processed/canto_profiles.jsonl"))
    parser.add_argument("--qdrant-mode", choices=["server", "local", "memory"], default=get_env("QDRANT_MODE", "server"))
    parser.add_argument("--qdrant-url", default=get_env("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--local-path", default=get_env("QDRANT_LOCAL_PATH", "qdrant_storage"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--embedding-backend", choices=["fastembed", "hashing"], default="fastembed")
    parser.add_argument("--output-dir", default="outputs/hierarchical_diagnostic")
    args = parser.parse_args()

    queries = load_queries(project_path(args.queries))
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever = HierarchicalRetriever(
        config_path=project_path(args.config),
        sqlite_path=project_path(args.sqlite),
        profiles_path=project_path(args.profiles),
        mode=args.qdrant_mode,
        qdrant_url=args.qdrant_url,
        local_path=args.local_path,
        embedding_backend=args.embedding_backend,
    )
    try:
        rows, chunk_metrics, canto_metrics = evaluate(retriever, queries, args.top_k)
    finally:
        retriever.close()

    details_path = output_dir / "hierarchical_results.csv"
    with details_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = output_dir / "hierarchical_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump([canto_metrics, chunk_metrics], fh, ensure_ascii=False, indent=2)

    print(canto_metrics)
    print(chunk_metrics)
    print(f"Resultados: {details_path}")
    print(f"Métricas: {metrics_path}")


if __name__ == "__main__":
    main()
