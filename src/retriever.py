from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from embedding_backends import make_embedding_backend
from common import SearchHit, get_env, lexical_terms, load_yaml, project_path
from qdrant_index import make_client


class HybridRetriever:
    def __init__(
        self,
        config_path: Path,
        sqlite_path: Path,
        mode: str = "server",
        qdrant_url: str = "http://localhost:6333",
        local_path: str = "qdrant_storage",
        embedding_backend: str = "fastembed",
    ) -> None:
        self.cfg = load_yaml(config_path)
        self.sqlite_path = sqlite_path
        self.client = make_client(mode, qdrant_url, local_path)
        self.embedding_backend = embedding_backend
        self.collection = self.cfg["collection_name"] if embedding_backend == "fastembed" else self.cfg["offline_collection_name"]
        self.embedder = make_embedding_backend(embedding_backend, self.cfg["embedding_model"], int(self.cfg["vector_size"]))

    def close(self) -> None:
        self.client.close()

    def _dense_search(self, query: str, limit: int, canto: int | None = None) -> list[SearchHit]:
        vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32).tolist()
        query_filter = None
        if canto is not None:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="canto", match=models.MatchValue(value=int(canto)))]
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(payload["chunk_id"]),
                    canto=int(payload["canto"]),
                    title=str(payload["titulo_canto"]),
                    text=str(payload["text"]),
                    score=float(point.score),
                    source="dense",
                    metadata=dict(payload),
                )
            )
        return hits

    def _lexical_search(self, query: str, limit: int, canto: int | None = None) -> list[SearchHit]:
        terms = lexical_terms(query)
        if not terms:
            return []

        base_sql = """
            SELECT f.chunk_id, c.canto_id, ca.titulo, c.text, bm25(chunks_fts) AS bm25_score,
                   c.chunk_index, c.paragraph_start, c.paragraph_end, c.word_count
            FROM chunks_fts AS f
            JOIN chunks AS c ON c.chunk_id = f.chunk_id
            JOIN cantos AS ca ON ca.canto_id = c.canto_id
            WHERE chunks_fts MATCH ?
        """
        if canto is not None:
            base_sql += " AND c.canto_id = ?"
        base_sql += " ORDER BY bm25_score ASC LIMIT ?"

        def run(match_query: str, row_limit: int) -> list[tuple]:
            params: list[Any] = [match_query]
            if canto is not None:
                params.append(int(canto))
            params.append(int(row_limit))
            with sqlite3.connect(self.sqlite_path) as con:
                return con.execute(base_sql, params).fetchall()

        # Primero exige todos los términos significativos. Si no hay suficientes resultados,
        # completa con una consulta OR menos restrictiva.
        def fts_token(term: str) -> str:
            if len(term) >= 6:
                prefix_len = max(5, len(term) - 2)
                return f"{term[:prefix_len]}*"
            return f'"{term}"'

        fts_terms = [fts_token(term) for term in terms]
        strict_query = " AND ".join(fts_terms)
        broad_query = " OR ".join(fts_terms)
        rows = run(strict_query, limit)
        if len(rows) < limit:
            seen = {row[0] for row in rows}
            for row in run(broad_query, limit * 3):
                if row[0] not in seen:
                    rows.append(row)
                    seen.add(row[0])
                if len(rows) >= limit:
                    break

        hits: list[SearchHit] = []
        for row in rows[:limit]:
            raw = float(row[4])
            score = 1.0 / (1.0 + abs(raw))
            hits.append(
                SearchHit(
                    chunk_id=str(row[0]),
                    canto=int(row[1]),
                    title=str(row[2]),
                    text=str(row[3]),
                    score=score,
                    source="lexical",
                    metadata={
                        "chunk_index": row[5],
                        "paragraph_start": row[6],
                        "paragraph_end": row[7],
                        "word_count": row[8],
                    },
                )
            )
        return hits

    def search(
        self,
        query: str,
        top_k: int | None = None,
        canto: int | None = None,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        rcfg = self.cfg["retrieval"]
        top_k = int(top_k or rcfg["final_top_k"])
        dense_limit = int(rcfg["dense_candidates"])
        lexical_limit = int(rcfg["lexical_candidates"])

        if mode == "dense":
            return self._dense_search(query, max(top_k, dense_limit), canto)[:top_k]
        if mode == "lexical":
            return self._lexical_search(query, max(top_k, lexical_limit), canto)[:top_k]
        if mode != "hybrid":
            raise ValueError("mode debe ser dense, lexical o hybrid")

        dense = self._dense_search(query, dense_limit, canto)
        lexical = self._lexical_search(query, lexical_limit, canto)
        rrf_k = float(rcfg["rrf_k"])
        weights = {
            "dense": float(rcfg["dense_weight"]),
            "lexical": float(rcfg["lexical_weight"]),
        }
        scores: dict[str, float] = defaultdict(float)
        by_id: dict[str, SearchHit] = {}
        sources: dict[str, set[str]] = defaultdict(set)

        for source, source_hits in (("dense", dense), ("lexical", lexical)):
            for rank, hit in enumerate(source_hits, start=1):
                scores[hit.chunk_id] += weights[source] / (rrf_k + rank)
                by_id.setdefault(hit.chunk_id, hit)
                sources[hit.chunk_id].add(source)

        ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
        result: list[SearchHit] = []
        for chunk_id in ordered:
            base = by_id[chunk_id]
            result.append(
                SearchHit(
                    chunk_id=base.chunk_id,
                    canto=base.canto,
                    title=base.title,
                    text=base.text,
                    score=float(scores[chunk_id]),
                    source="+".join(sorted(sources[chunk_id])),
                    metadata=base.metadata,
                )
            )
        return result

    def structured_context(self, canto_ids: list[int]) -> dict[str, Any]:
        canto_ids = sorted(set(int(x) for x in canto_ids))
        if not canto_ids:
            return {"cantos": []}
        placeholders = ",".join("?" for _ in canto_ids)
        with sqlite3.connect(self.sqlite_path) as con:
            con.row_factory = sqlite3.Row
            cantos = [dict(r) for r in con.execute(
                f"SELECT canto_id, titulo, resumen, uso FROM cantos WHERE canto_id IN ({placeholders}) ORDER BY canto_id",
                canto_ids,
            )]
            for canto in cantos:
                cid = canto["canto_id"]
                canto["events"] = [r[0] for r in con.execute(
                    "SELECT description FROM events WHERE canto_id=? ORDER BY event_order", (cid,)
                )]
                canto["characters"] = [r[0] for r in con.execute(
                    "SELECT DISTINCT surface_name FROM canto_characters WHERE canto_id=? ORDER BY surface_name", (cid,)
                )]
                canto["places"] = [r[0] for r in con.execute(
                    "SELECT p.name FROM canto_places cp JOIN places p ON p.place_id=cp.place_id WHERE cp.canto_id=? ORDER BY p.name", (cid,)
                )]
                canto["themes"] = [r[0] for r in con.execute(
                    "SELECT t.name FROM canto_themes ct JOIN themes t ON t.theme_id=ct.theme_id WHERE ct.canto_id=? ORDER BY t.name", (cid,)
                )]
                canto["relations"] = [r[0] for r in con.execute(
                    "SELECT relation_text FROM relations WHERE canto_id=? ORDER BY relation_id", (cid,)
                )]
        return {"cantos": cantos}


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta densa, léxica o híbrida sobre La Odisea.")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["dense", "lexical", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--canto", type=int)
    parser.add_argument("--config", default="config/retrieval.yaml")
    parser.add_argument("--sqlite", default=get_env("SQLITE_PATH", "data/processed/corpus_odisea.sqlite"))
    parser.add_argument("--qdrant-mode", choices=["server", "local", "memory"], default=get_env("QDRANT_MODE", "server"))
    parser.add_argument("--qdrant-url", default=get_env("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--local-path", default=get_env("QDRANT_LOCAL_PATH", "qdrant_storage"))
    parser.add_argument("--embedding-backend", choices=["fastembed", "hashing"], default="fastembed")
    args = parser.parse_args()

    retriever = HybridRetriever(
        config_path=project_path(args.config),
        sqlite_path=project_path(args.sqlite),
        mode=args.qdrant_mode,
        qdrant_url=args.qdrant_url,
        local_path=args.local_path,
        embedding_backend=args.embedding_backend,
    )
    try:
        hits = retriever.search(args.query, top_k=args.top_k, canto=args.canto, mode=args.mode)
        for rank, hit in enumerate(hits, start=1):
            snippet = hit.text.replace("\n", " ")[:240]
            print(f"{rank}. {hit.chunk_id} | canto {hit.canto} | {hit.score:.6f} | {hit.source}")
            print(f"   {hit.title}")
            print(f"   {snippet}...")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
