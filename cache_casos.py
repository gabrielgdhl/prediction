from pathlib import Path
import hashlib
import json
import pickle
import time


ROOT = Path(__file__).resolve().parent

CACHE_DIR = (
    ROOT
    / ".cache"
    / "casos"
)

CACHE_CASOS = (
    CACHE_DIR
    / "casos_walk_forward.pkl"
)

CACHE_METADATA = (
    CACHE_DIR
    / "casos_walk_forward_metadata.json"
)


# ============================================================
# VERSÃO MANUAL
#
# Se mudarmos a lógica de gerar_caso(), incrementar.
# ============================================================

CACHE_CASOS_VERSION = 1


# ============================================================
# HASH
# ============================================================

def calcular_hash_arquivo(
    caminho
):
    caminho = Path(
        caminho
    )

    sha256 = (
        hashlib.sha256()
    )

    with open(
        caminho,
        "rb"
    ) as arquivo:

        while True:

            bloco = arquivo.read(
                1024 * 1024
            )

            if not bloco:
                break

            sha256.update(
                bloco
            )

    return (
        sha256.hexdigest()
    )


# ============================================================
# METADATA
# ============================================================

def gerar_metadata(
    arquivos_dependencia,
    parametros
):

    hashes = {}

    for nome, caminho in (
        arquivos_dependencia.items()
    ):

        hashes[
            nome
        ] = (
            calcular_hash_arquivo(
                caminho
            )
        )

    return {
        "cache_version":
            CACHE_CASOS_VERSION,

        "hashes":
            hashes,

        "parametros":
            parametros,
    }


# ============================================================
# COMPARAÇÃO
# ============================================================

def cache_valido(
    arquivos_dependencia,
    parametros
):

    if (
        not CACHE_CASOS.exists()
        or not CACHE_METADATA.exists()
    ):
        return False

    try:

        with open(
            CACHE_METADATA,
            "r",
            encoding="utf-8"
        ) as arquivo:

            metadata_salva = (
                json.load(
                    arquivo
                )
            )

    except (
        OSError,
        json.JSONDecodeError
    ):
        return False

    metadata_atual = (
        gerar_metadata(
            arquivos_dependencia=
                arquivos_dependencia,

            parametros=
                parametros,
        )
    )

    return (
        metadata_salva
        == metadata_atual
    )


# ============================================================
# CARREGAR
# ============================================================

def carregar_cache_casos(
    arquivos_dependencia,
    parametros
):

    if not cache_valido(
        arquivos_dependencia=
            arquivos_dependencia,

        parametros=
            parametros,
    ):

        return None

    print()
    print(
        "=" * 70
    )
    print(
        "CACHE DE CASOS ENCONTRADO"
    )
    print(
        "=" * 70
    )

    inicio = (
        time.time()
    )

    with open(
        CACHE_CASOS,
        "rb"
    ) as arquivo:

        payload = (
            pickle.load(
                arquivo
            )
        )

    print(
        f"Casos carregados em "
        f"{time.time() - inicio:.2f}s"
    )

    print(
        f"Casos: "
        f"{len(payload['casos'])}"
    )

    return payload


# ============================================================
# SALVAR
# ============================================================

def salvar_cache_casos(
    casos,
    total,
    primeiro_teste,
    arquivos_dependencia,
    parametros,
):

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    payload = {
        "casos":
            casos,

        "total":
            total,

        "primeiro_teste":
            primeiro_teste,
    }

    metadata = (
        gerar_metadata(
            arquivos_dependencia=
                arquivos_dependencia,

            parametros=
                parametros,
        )
    )

    caminho_temporario = (
        CACHE_CASOS.with_suffix(
            ".tmp"
        )
    )

    print()
    print(
        "Salvando cache de casos..."
    )

    inicio = (
        time.time()
    )

    # Primeiro escreve arquivo temporário.
    # Assim uma interrupção não destrói
    # um cache anterior válido.

    with open(
        caminho_temporario,
        "wb"
    ) as arquivo:

        pickle.dump(
            payload,
            arquivo,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    caminho_temporario.replace(
        CACHE_CASOS
    )

    with open(
        CACHE_METADATA,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            metadata,
            arquivo,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Cache salvo em "
        f"{time.time() - inicio:.2f}s"
    )

    print(
        CACHE_CASOS
    )


# ============================================================
# LIMPAR
# ============================================================

def limpar_cache_casos():

    for caminho in (
        CACHE_CASOS,
        CACHE_METADATA,
    ):

        if caminho.exists():
            caminho.unlink()