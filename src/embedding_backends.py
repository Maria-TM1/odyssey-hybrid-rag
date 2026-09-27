from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    name: str
    vector_size: int

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]: ...
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass
class FastEmbedBackend:
    model_name: str
    vector_size: int
    name: str = "fastembed"

    def __post_init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=self.model_name)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(texts, batch_size=batch_size)]

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(self._model.query_embed(text)), dtype=np.float32)


@dataclass
class HashingBackend:
    """Backend determinista sin descargas para validar el pipeline.

    No sustituye al modelo semántico del experimento final. Usa hashing de unigramas
    y bigramas en un espacio fijo de 384 dimensiones.
    """

    vector_size: int
    name: str = "hashing"

    def __post_init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vectorizer = HashingVectorizer(
            n_features=self.vector_size,
            alternate_sign=False,
            norm="l2",
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
        )

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        matrix = self._vectorizer.transform(texts)
        return [row.toarray().ravel().astype(np.float32) for row in matrix]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vectorizer.transform([text]).toarray().ravel().astype(np.float32)


def make_embedding_backend(kind: str, model_name: str, vector_size: int) -> EmbeddingBackend:
    if kind == "fastembed":
        return FastEmbedBackend(model_name=model_name, vector_size=vector_size)
    if kind == "hashing":
        return HashingBackend(vector_size=vector_size)
    raise ValueError("embedding_backend debe ser fastembed o hashing")
