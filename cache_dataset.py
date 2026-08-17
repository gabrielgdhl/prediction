from pathlib import Path
import hashlib
import json
import time

import numpy as np


ROOT = Path(__file__).resolve().parent

CACHE_DIR = (
    ROOT
    / "cache"
)

CACHE_DATASET = (
    CACHE_DIR
    / "features_v2_dataset.npz"
)

CACHE_METADATA = (
    CACHE_DIR
    / "features_v2_dataset.json"
)


# ============================================================
# HASH DA BASE
# ============================================================

def calcular_hash_arquivo(
    caminho
):
    """
    Gera uma assinatura do arquivo Excel.

    Se qualquer coisa mudar no arquivo,
    o hash também muda e invalidamos o cache.
    """

    sha256 = hashlib.sha256()

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
# VALIDAR CACHE
# ============================================================

def cache_valido(
    caminho_excel,
    janela_minima
):
    if not CACHE_DATASET.exists():
        return False

    if not CACHE_METADATA.exists():
        return False

    try:

        with open(
            CACHE_METADATA,
            "r",
            encoding="utf-8"
        ) as arquivo:

            metadata = json.load(
                arquivo
            )

    except Exception:

        return False

    hash_atual = (
        calcular_hash_arquivo(
            caminho_excel
        )
    )

    if (
        metadata.get(
            "hash_excel"
        )
        != hash_atual
    ):
        print(
            "Cache V2 inválido: "
            "arquivo Excel mudou."
        )

        return False

    if (
        metadata.get(
            "janela_minima"
        )
        != janela_minima
    ):
        print(
            "Cache V2 inválido: "
            "janela mínima mudou."
        )

        return False

    return True


# ============================================================
# SALVAR
# ============================================================

def salvar_cache(
    caminho_excel,
    janela_minima,
    X,
    y,
    indices_target,
    dezenas,
    matriz_binaria
):
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "Salvando dataset V2 em cache..."
    )

    inicio = time.time()

    np.savez_compressed(
        CACHE_DATASET,

        X=X,

        y=y,

        indices_target=
            indices_target,

        dezenas=
            dezenas,

        matriz_binaria=
            matriz_binaria
    )

    metadata = {
        "hash_excel":
            calcular_hash_arquivo(
                caminho_excel
            ),

        "janela_minima":
            janela_minima,

        "shape_X":
            list(
                X.shape
            ),

        "shape_y":
            list(
                y.shape
            )
    }

    with open(
        CACHE_METADATA,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            metadata,
            arquivo,
            indent=2
        )

    print(
        f"Cache salvo em "
        f"{time.time() - inicio:.2f}s"
    )

    print(
        CACHE_DATASET
    )


# ============================================================
# CARREGAR
# ============================================================

def carregar_cache():

    print()
    print(
        "Carregando dataset V2 do cache..."
    )

    inicio = time.time()

    dados = np.load(
        CACHE_DATASET,
        allow_pickle=False
    )

    X = dados[
        "X"
    ]

    y = dados[
        "y"
    ]

    indices_target = dados[
        "indices_target"
    ]

    dezenas = dados[
        "dezenas"
    ]

    matriz_binaria = dados[
        "matriz_binaria"
    ]

    print(
        f"Cache carregado em "
        f"{time.time() - inicio:.2f}s"
    )

    print(
        f"X = {X.shape}"
    )

    return (
        X,
        y,
        indices_target,
        dezenas,
        matriz_binaria
    )


# ============================================================
# OBTER OU CRIAR
# ============================================================

def obter_dataset_v2(
    caminho_excel,
    df_bolas,
    classe_gerador,
    janela_minima
):
    """
    Interface principal.

    Se existe cache válido:
        carrega.

    Senão:
        gera tudo,
        salva,
        retorna.
    """

    if cache_valido(
        caminho_excel,
        janela_minima
    ):

        (
            X,
            y,
            indices_target,
            dezenas,
            matriz_binaria
        ) = carregar_cache()

        return (
            None,
            X,
            y,
            indices_target,
            dezenas,
            matriz_binaria
        )

    # ========================================================
    # CACHE NÃO EXISTE
    # ========================================================

    print()
    print(
        "Cache V2 não encontrado "
        "ou inválido."
    )

    print(
        "Gerando dataset..."
    )

    gerador = classe_gerador(
        df_bolas
    )

    inicio = time.time()

    (
        X,
        y,
        indices_target,
        dezenas
    ) = gerador.construir_dataset(
        janela_minima=
            janela_minima
    )

    print(
        f"Dataset calculado em "
        f"{time.time() - inicio:.1f}s"
    )

    salvar_cache(
        caminho_excel=
            caminho_excel,

        janela_minima=
            janela_minima,

        X=X,

        y=y,

        indices_target=
            indices_target,

        dezenas=
            dezenas,

        matriz_binaria=
            gerador.matriz_binaria
    )

    return (
        gerador,
        X,
        y,
        indices_target,
        dezenas,
        gerador.matriz_binaria
    )