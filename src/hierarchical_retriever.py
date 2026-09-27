from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import models

from common import SearchHit, get_env, lexical_terms, load_jsonl, load_yaml, project_path
from embedding_backends import make_embedding_backend
from qdrant_index import make_client
from retriever import HybridRetriever


@dataclass(frozen=True)
class CantoHit:
    canto: int
    title: str
    score: float
    source: str
    metadata: dict[str, Any]


class HierarchicalRetriever:
    """Recuperador jerárquico para La Odisea.

    Nivel 1: selecciona cantos candidatos mediante perfiles enriquecidos.
    Nivel 2: busca chunks dentro de los cantos candidatos y reordena los resultados.
    """

    def __init__(
        self,
        config_path: Path,
        sqlite_path: Path,
        profiles_path: Path,
        mode: str = "server",
        qdrant_url: str = "http://localhost:6333",
        local_path: str = "qdrant_storage",
        embedding_backend: str = "fastembed",
    ) -> None:
        self.cfg = load_yaml(config_path)
        self.profiles = load_jsonl(profiles_path)
        if not self.profiles:
            raise ValueError(f"No se han encontrado perfiles de canto en {profiles_path}")
        self.profile_by_canto = {int(p["canto"]): p for p in self.profiles}
        self.client = make_client(mode, qdrant_url, local_path)
        self.embedding_backend = embedding_backend
        self.canto_collection = (
            self.cfg["canto_collection_name"]
            if embedding_backend == "fastembed"
            else self.cfg.get("offline_canto_collection_name", "odisea_canto_profiles_v1_hashing")
        )
        self.embedder = make_embedding_backend(
            embedding_backend,
            self.cfg["embedding_model"],
            int(self.cfg["vector_size"]),
        )
        self.chunk_retriever = HybridRetriever(
            config_path=config_path,
            sqlite_path=sqlite_path,
            mode=mode,
            qdrant_url=qdrant_url,
            local_path=local_path,
            embedding_backend=embedding_backend,
        )

    def close(self) -> None:
        self.chunk_retriever.close()
        self.client.close()

    @staticmethod
    def _normalise_scores(hits: list[CantoHit]) -> list[CantoHit]:
        if not hits:
            return []
        max_score = max(h.score for h in hits) or 1.0
        return [
            CantoHit(h.canto, h.title, float(h.score) / max_score, h.source, h.metadata)
            for h in hits
        ]

    def _dense_canto_search(self, query: str, limit: int) -> list[CantoHit]:
        vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32).tolist()
        response = self.client.query_points(
            collection_name=self.canto_collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        hits: list[CantoHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                CantoHit(
                    canto=int(payload["canto"]),
                    title=str(payload.get("titulo", "")),
                    score=float(point.score),
                    source="canto_dense",
                    metadata=dict(payload),
                )
            )
        return hits

    def _lexical_canto_search(self, query: str, limit: int) -> list[CantoHit]:
        q_terms = lexical_terms(query)
        if not q_terms:
            return []
        q_counter = Counter(q_terms)
        rows: list[CantoHit] = []
        for profile in self.profiles:
            text = str(profile.get("text", ""))
            terms = lexical_terms(text)
            if not terms:
                continue
            profile_counter = Counter(terms)
            score = 0.0
            matched_terms: list[str] = []
            for term, q_count in q_counter.items():
                # Coincidencia exacta y coincidencia por prefijo para variantes flexivas.
                exact = profile_counter.get(term, 0)
                prefix = 0
                if len(term) >= 6:
                    prefix_key = term[: max(5, len(term) - 2)]
                    prefix = sum(v for k, v in profile_counter.items() if k.startswith(prefix_key))
                freq = max(exact, prefix)
                if freq > 0:
                    matched_terms.append(term)
                    # IDF suave sobre 24 cantos + saturación logarítmica de frecuencia.
                    df = sum(1 for p in self.profiles if term in set(lexical_terms(str(p.get("text", "")))))
                    idf = math.log((1 + len(self.profiles)) / (1 + df)) + 1.0
                    score += q_count * idf * (1.0 + math.log(freq))
            if score > 0:
                rows.append(
                    CantoHit(
                        canto=int(profile["canto"]),
                        title=str(profile.get("titulo", "")),
                        score=score,
                        source="canto_lexical",
                        metadata={**profile, "matched_terms": matched_terms},
                    )
                )
        rows.sort(key=lambda h: h.score, reverse=True)
        return self._normalise_scores(rows[:limit])

    def select_cantos(self, query: str, top_k: int | None = None) -> list[CantoHit]:
        hcfg = self.cfg["hierarchical"]
        top_k = int(top_k or hcfg["canto_top_k"])
        dense_limit = int(hcfg["canto_dense_candidates"])
        lexical_limit = int(hcfg["canto_lexical_candidates"])
        dense = self._dense_canto_search(query, dense_limit)
        lexical = self._lexical_canto_search(query, lexical_limit)
        rrf_k = float(hcfg["canto_rrf_k"])
        weights = {
            "canto_dense": float(hcfg["canto_dense_weight"]),
            "canto_lexical": float(hcfg["canto_lexical_weight"]),
        }
        scores: dict[int, float] = defaultdict(float)
        by_canto: dict[int, CantoHit] = {}
        sources: dict[int, set[str]] = defaultdict(set)
        metadata_by_canto: dict[int, dict[str, Any]] = defaultdict(dict)
        for source, source_hits in (("canto_dense", dense), ("canto_lexical", lexical)):
            for rank, hit in enumerate(source_hits, start=1):
                scores[hit.canto] += weights[source] / (rrf_k + rank)
                by_canto.setdefault(hit.canto, hit)
                sources[hit.canto].add(source)
                metadata_by_canto[hit.canto][source] = {
                    "rank": rank,
                    "raw_score": hit.score,
                    "metadata": hit.metadata,
                }
        ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
        result: list[CantoHit] = []
        for canto in ordered:
            profile = self.profile_by_canto.get(canto, {})
            base = by_canto[canto]
            result.append(
                CantoHit(
                    canto=canto,
                    title=str(profile.get("titulo") or base.title),
                    score=float(scores[canto]),
                    source="+".join(sorted(sources[canto])),
                    metadata={
                        "profile": profile,
                        "profile_sources": metadata_by_canto[canto],
                    },
                )
            )
        return result

    def search(self, query: str, top_k: int | None = None, canto_top_k: int | None = None) -> list[SearchHit]:
        hcfg = self.cfg["hierarchical"]
        top_k = int(top_k or hcfg["final_top_k"])
        chunks_per_canto = int(hcfg["chunks_per_canto"])
        canto_hits = self.select_cantos(query, top_k=canto_top_k)
        by_chunk: dict[str, SearchHit] = {}
        combined_scores: dict[str, float] = {}
        selected_cantos = [h.canto for h in canto_hits]
        max_canto_score = max((h.score for h in canto_hits), default=1.0) or 1.0
        canto_score_by_id = {h.canto: h.score / max_canto_score for h in canto_hits}
        canto_source_by_id = {h.canto: h.source for h in canto_hits}

        for canto_hit in canto_hits:
            chunk_hits = self.chunk_retriever.search(
                query,
                top_k=chunks_per_canto,
                canto=canto_hit.canto,
                mode="hybrid",
            )
            for rank, hit in enumerate(chunk_hits, start=1):
                # La puntuación de chunk procede de RRF y puede ser pequeña. Se usa también
                # el rango para que el primer chunk de un canto candidato no quede penalizado.
                chunk_rank_score = 1.0 / rank
                score = 0.60 * canto_score_by_id[canto_hit.canto] + 0.40 * chunk_rank_score
                previous = combined_scores.get(hit.chunk_id)
                if previous is None or score > previous:
                    combined_scores[hit.chunk_id] = score
                    by_chunk[hit.chunk_id] = SearchHit(
                        chunk_id=hit.chunk_id,
                        canto=hit.canto,
                        title=hit.title,
                        text=hit.text,
                        score=float(score),
                        source=f"hierarchical:{canto_source_by_id[canto_hit.canto]}:{hit.source}",
                        metadata={
                            **hit.metadata,
                            "selected_cantos": selected_cantos,
                            "canto_profile_score": canto_score_by_id[canto_hit.canto],
                            "canto_profile_source": canto_source_by_id[canto_hit.canto],
                            "chunk_rank_within_canto": rank,
                        },
                    )
        ordered = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        return [by_chunk[chunk_id] for chunk_id in ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recuperación jerárquica: perfiles de canto + chunks.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--canto-top-k", type=int)
    parser.add_argument("--config", default="config/retrieval_hierarchical.yaml")
    parser.add_argument("--sqlite", default=get_env("SQLITE_PATH", "data/processed/corpus_odisea.sqlite"))
    parser.add_argument("--profiles", default=get_env("CANTO_PROFILES_PATH", "data/processed/canto_profiles.jsonl"))
    parser.add_argument("--qdrant-mode", choices=["server", "local", "memory"], default=get_env("QDRANT_MODE", "server"))
    parser.add_argument("--qdrant-url", default=get_env("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--local-path", default=get_env("QDRANT_LOCAL_PATH", "qdrant_storage"))
    parser.add_argument("--embedding-backend", choices=["fastembed", "hashing"], default="fastembed")
    args = parser.parse_args()

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
        cantos = retriever.select_cantos(args.query, top_k=args.canto_top_k)
        print("Cantos seleccionados:")
        for i, hit in enumerate(cantos, start=1):
            print(f"  {i}. Canto {hit.canto}: {hit.title} | score={hit.score:.6f} | {hit.source}")
        print("\nChunks recuperados:")
        hits = retriever.search(args.query, top_k=args.top_k, canto_top_k=args.canto_top_k)
        for i, hit in enumerate(hits, start=1):
            snippet = " ".join(hit.text.split()[:70])
            print(f"[{i}] canto={hit.canto} chunk={hit.chunk_id} score={hit.score:.6f} source={hit.source}")
            print(f"    {snippet}...")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
