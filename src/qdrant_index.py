from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from qdrant_client import QdrantClient, models

from embedding_backends import make_embedding_backend
from common import get_env, load_jsonl, load_yaml, project_path, qdrant_point_id


def make_client(mode: str, url: str, local_path: str) -> QdrantClient:
    if mode == "local":
        return QdrantClient(path=str(project_path(local_path)))
    if mode == "memory":
        return QdrantClient(":memory:")
    return QdrantClient(url=url, timeout=60)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int, recreate: bool) -> None:
    exists = client.collection_exists(collection)
    if exists and recreate:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        client.create_payload_index(collection, "canto", models.PayloadSchemaType.INTEGER)
        client.create_payload_index(collection, "chunk_id", models.PayloadSchemaType.KEYWORD)


def batched(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_index(
    chunks_path: Path,
    config_path: Path,
    mode: str,
    url: str,
    local_path: str,
    recreate: bool,
    batch_size: int,
    embedding_backend: str,
) -> dict[str, int | str]:
    cfg = load_yaml(config_path)
    collection = cfg["collection_name"] if embedding_backend == "fastembed" else cfg["offline_collection_name"]
    model_name = cfg["embedding_model"]
    vector_size = int(cfg["vector_size"])

    rows = load_jsonl(chunks_path)
    if not rows:
        raise ValueError("No se encontraron chunks para indexar.")

    client = make_client(mode, url, local_path)
    ensure_collection(client, collection, vector_size, recreate)
    embedder = make_embedding_backend(embedding_backend, model_name, vector_size)

    uploaded = 0
    for batch in batched(rows, batch_size):
        texts = [row["text"] for row in batch]
        vectors = embedder.embed_documents(texts, batch_size=min(batch_size, 32))
        points: list[models.PointStruct] = []
        for row, vector in zip(batch, vectors, strict=True):
            vector_array = np.asarray(vector, dtype=np.float32)
            if vector_array.shape[0] != vector_size:
                raise ValueError(
                    f"Dimensión inesperada para {row['chunk_id']}: "
                    f"{vector_array.shape[0]} != {vector_size}"
                )
            payload = {
                "chunk_id": row["chunk_id"],
                "canto": int(row["canto"]),
                "titulo_canto": row["titulo_canto"],
                "chunk_index": int(row["chunk_index"]),
                "paragraph_start": row.get("paragraph_start"),
                "paragraph_end": row.get("paragraph_end"),
                "word_count": int(row["word_count"]),
                "text": row["text"],
                "obra": row.get("source_work", "La Odisea"),
                "traductor": row.get("source_translator", "Luis Segalá y Estalella"),
                "edicion": row.get("source_edition_year", 1910),
            }
            points.append(
                models.PointStruct(
                    id=qdrant_point_id(row["chunk_id"]),
                    vector=vector_array.tolist(),
                    payload=payload,
                )
            )
        client.upsert(collection_name=collection, points=points, wait=True)
        uploaded += len(points)
        print(f"Indexados {uploaded}/{len(rows)} chunks")

    count = client.count(collection_name=collection, exact=True).count
    client.close()
    return {"collection": collection, "backend": embedding_backend, "model": model_name, "indexed": int(count)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera embeddings e indexa los chunks de La Odisea en Qdrant.")
    parser.add_argument("--config", default="config/retrieval.yaml")
    parser.add_argument("--chunks", default=get_env("CHUNKS_PATH", "data/processed/chunks.jsonl"))
    parser.add_argument("--mode", choices=["server", "local", "memory"], default=get_env("QDRANT_MODE", "server"))
    parser.add_argument("--url", default=get_env("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--local-path", default=get_env("QDRANT_LOCAL_PATH", "qdrant_storage"))
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-backend", choices=["fastembed", "hashing"], default="fastembed")
    args = parser.parse_args()

    report = build_index(
        chunks_path=project_path(args.chunks),
        config_path=project_path(args.config),
        mode=args.mode,
        url=args.url,
        local_path=args.local_path,
        recreate=args.recreate,
        batch_size=args.batch_size,
        embedding_backend=args.embedding_backend,
    )
    print(report)


if __name__ == "__main__":
    main()
