"""Cache persistente, versionado e atômico para artefatos causais.

Somente casos walk-forward e matrizes meta são persistidos. Modelos e
previsões/resultados nunca são gravados neste cache.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CACHE_VERSION = 2


def hash_arquivo(caminho: Path) -> str:
    sha = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            sha.update(bloco)
    return sha.hexdigest()


def assinatura(dependencias: Mapping[str, Path], parametros: Mapping[str, Any]) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "dependencias": {
            nome: hash_arquivo(Path(caminho))
            for nome, caminho in sorted(dependencias.items())
        },
        "parametros": json.loads(json.dumps(parametros, sort_keys=True)),
    }


class CacheExperimento:
    def __init__(self, raiz: Path, nome: str, assinatura_atual: dict):
        self.diretorio = Path(raiz) / ".cache" / nome
        self.metadata = self.diretorio / "metadata.json"
        self.casos_path = self.diretorio / "casos.pkl"
        self.meta_path = self.diretorio / "matrizes_meta.npz"
        self.assinatura_atual = assinatura_atual

    def valido(self) -> bool:
        try:
            with self.metadata.open("r", encoding="utf-8") as arquivo:
                return json.load(arquivo) == self.assinatura_atual
        except (OSError, json.JSONDecodeError):
            return False

    def _preparar(self) -> None:
        self.diretorio.mkdir(parents=True, exist_ok=True)
        temporario = self.metadata.with_suffix(".tmp")
        temporario.write_text(
            json.dumps(self.assinatura_atual, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporario.replace(self.metadata)

    def carregar_casos(self) -> dict | None:
        if not self.valido() or not self.casos_path.exists():
            return None
        try:
            with self.casos_path.open("rb") as arquivo:
                payload = pickle.load(arquivo)
            return payload if isinstance(payload, dict) else None
        except (OSError, EOFError, pickle.UnpicklingError):
            return None

    def salvar_casos(self, payload: dict) -> None:
        self._preparar()
        temporario = self.casos_path.with_suffix(".tmp")
        with temporario.open("wb") as arquivo:
            pickle.dump(payload, arquivo, protocol=pickle.HIGHEST_PROTOCOL)
        temporario.replace(self.casos_path)

    def carregar_meta(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not self.valido() or not self.meta_path.exists():
            return None
        try:
            with np.load(self.meta_path, allow_pickle=False) as dados:
                return (
                    dados["indices"].copy(),
                    dados["X_meta"].copy(),
                    dados["y_meta"].copy(),
                )
        except (OSError, ValueError, KeyError):
            return None

    def salvar_meta(self, indices: np.ndarray, X_meta: np.ndarray, y_meta: np.ndarray) -> None:
        self._preparar()
        temporario = self.meta_path.with_name(self.meta_path.name + ".tmp.npz")
        np.savez(temporario, indices=indices, X_meta=X_meta, y_meta=y_meta)
        temporario.replace(self.meta_path)

