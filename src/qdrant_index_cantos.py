from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
from qdrant_client import QdrantClient, models

from embedding_backends import make_embedding_backend
from common import get_env, load_jsonl, load_yaml, project_path
from qdrant_index import make_client


def profile_point_id(profile_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tfm-odisea-canto-profile:{profile_id}"))


def ensure_canto_collection(client: QdrantClient, collection: str, vector_size: int, recreate: bool) -> None:
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
        client.create_payload_index(collection, "profile_id", models.PayloadSchemaType.KEYWORD)


def batched(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_canto_index(
    profiles_path: Path,
    config_path: Path,
    mode: str,
    url: str,
    local_path: str,
    recreate: bool,
    batch_size: int,
    embedding_backend: str,
) -> dict[str, int | str]:
    cfg = load_yaml(config_path)
    collection = (
        cfg["canto_collection_name"]
        if embedding_backend == "fastembed"
        else cfg.get("offline_canto_collection_name", "odisea_canto_profiles_v1_hashing")
    )
    model_name = cfg["embedding_model"]
    vector_size = int(cfg["vector_size"])

    rows = load_jsonl(profiles_path)
    if not rows:
        raise ValueError("No se encontraron perfiles de canto para indexar.")
    if len(rows) != 24:
        print(f"ADVERTENCIA: se esperaban 24 perfiles y se han encontrado {len(rows)}.")

    client = make_client(mode, url, local_path)
    ensure_canto_collection(client, collection, vector_size, recreate)
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
                    f"Dimensión inesperada para {row['profile_id']}: "
                    f"{vector_array.shape[0]} != {vector_size}"
                )
            payload = {
                "profile_id": row["profile_id"],
                "canto": int(row["canto"]),
                "titulo": row.get("titulo", ""),
                "resumen": row.get("resumen", ""),
                "personajes": row.get("personajes", []),
                "alias": row.get("alias", []),
                "lugares": row.get("lugares", []),
                "eventos": row.get("eventos", []),
                "temas": row.get("temas", []),
                "relaciones": row.get("relaciones", []),
                "palabras_clave": row.get("palabras_clave", []),
                "uso": row.get("uso", ""),
                "text": row["text"],
            }
            points.append(
                models.PointStruct(
                    id=profile_point_id(str(row["profile_id"])),
                    vector=vector_array.tolist(),
                    payload=payload,
                )
            )
        client.upsert(collection_name=collection, points=points, wait=True)
        uploaded += len(points)
        print(f"Indexados {uploaded}/{len(rows)} perfiles de canto")

    count = client.count(collection_name=collection, exact=True).count
    client.close()
    return {"collection": collection, "backend": embedding_backend, "model": model_name, "indexed": int(count)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa los 24 perfiles enriquecidos de canto en Qdrant.")
    parser.add_argument("--config", default="config/retrieval_hierarchical.yaml")
    parser.add_argument("--profiles", default=get_env("CANTO_PROFILES_PATH", "data/processed/canto_profiles.jsonl"))
    parser.add_argument("--mode", choices=["server", "local", "memory"], default=get_env("QDRANT_MODE", "server"))
    parser.add_argument("--url", default=get_env("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--local-path", default=get_env("QDRANT_LOCAL_PATH", "qdrant_storage"))
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--embedding-backend", choices=["fastembed", "hashing"], default="fastembed")
    args = parser.parse_args()

    report = build_canto_index(
        profiles_path=project_path(args.profiles),
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
