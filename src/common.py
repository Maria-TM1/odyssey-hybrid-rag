from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en {path}, línea {line_no}: {exc}") from exc
    return rows


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tfm-odisea:{chunk_id}"))


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_env(name: str, default: str) -> str:
    return os.getenv(name, default)


SPANISH_STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra",
    "cual", "cuando", "de", "del", "desde", "donde", "durante", "e", "el", "ella", "ellas",
    "ellos", "en", "entre", "era", "eran", "es", "esa", "ese", "eso", "esta", "estaba",
    "están", "este", "esto", "fue", "ha", "hacia", "hasta", "hay", "la", "las", "le", "les",
    "lo", "los", "más", "me", "mi", "mientras", "muy", "ni", "no", "o", "para", "pero",
    "por", "porque", "que", "qué", "se", "sin", "sobre", "su", "sus", "también", "te", "tu",
    "un", "una", "uno", "unos", "y", "ya"
}


def lexical_terms(query: str) -> list[str]:
    tokens = re.findall(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", query.lower())
    return [t for t in tokens if len(t) >= 3 and t not in SPANISH_STOPWORDS]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    canto: int
    title: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any]
