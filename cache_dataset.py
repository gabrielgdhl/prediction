from pathlib import Path

import hashlib
import json
import time

import numpy as np


# ============================================================
# PATHS
# ============================================================

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
    / "features_v2_metadata.json"
)


# ============================================================
# VERSÃO MANUAL DO CACHE
#
# Se quisermos invalidar tudo manualmente no futuro,
# basta incrementar esse número.
# ============================================================

CACHE_VERSION = 1


# ============================================================
# HASH DE ARQUIVO
# ============================================================

def calcular_hash_arquivo(
    caminho
):
    """
    Calcula SHA256 do arquivo.

    Usamos isso para detectar alterações em:

        - lotofacil_resultados.xlsx
        - features_v2_reference.py
    """

    caminho = Path(
        caminho
    )

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

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
# METADATA ESPERADA
# ============================================================

def gerar_metadata_esperada(
    caminho_excel,
    caminho_features,
    janela_minima
):
    """
    Tudo que influencia o conteúdo do dataset
    precisa fazer parte da validação do cache.
    """

    return {
        "cache_version":
            CACHE_VERSION,

        "hash_excel":
            calcular_hash_arquivo(
                caminho_excel
            ),

        "hash_features":
            calcular_hash_arquivo(
                caminho_features
            ),

        "janela_minima":
            int(
                janela_minima
            )
    }


# ============================================================
# CARREGAR METADATA
# ============================================================

def carregar_metadata():
    if not CACHE_METADATA.exists():
        return None

    try:
        with open(
            CACHE_METADATA,
            "r",
            encoding="utf-8"
        ) as arquivo:

            return json.load(
                arquivo
            )

    except (
        OSError,
        json.JSONDecodeError
    ):
        return None


# ============================================================
# VALIDAR CACHE
# ============================================================

def cache_valido(
    caminho_excel,
    caminho_features,
    janela_minima
):
    """
    Cache só é considerado válido se:

        1. arquivo NPZ existir
        2. metadata existir
        3. versão for igual
        4. Excel for exatamente o mesmo
        5. features_v2_reference.py for exatamente o mesmo
        6. janela_minima for igual
    """

    if not CACHE_DATASET.exists():

        print(
            "Cache V2: dataset ainda não existe."
        )

        return False

    metadata_salva = (
        carregar_metadata()
    )

    if metadata_salva is None:

        print(
            "Cache V2: metadata inexistente ou inválida."
        )

        return False

    metadata_esperada = (
        gerar_metadata_esperada(
            caminho_excel=
                caminho_excel,

            caminho_features=
                caminho_features,

            janela_minima=
                janela_minima
        )
    )

    for chave, valor_esperado in (
        metadata_esperada.items()
    ):

        valor_salvo = (
            metadata_salva.get(
                chave
            )
        )

        if (
            valor_salvo
            != valor_esperado
        ):

            print(
                f"Cache V2 inválido: "
                f"'{chave}' mudou."
            )

            return False

    return True


# ============================================================
# SALVAR CACHE
# ============================================================

def salvar_cache(
    caminho_excel,
    caminho_features,
    janela_minima,
    X,
    y,
    indices_target,
    dezenas,
    matriz_binaria
):
    """
    Salva dataset e informações necessárias
    para reconstruir o estado usado pelos backtests.
    """

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "Salvando dataset V2 em cache..."
    )

    inicio = time.time()

    # --------------------------------------------------------
    # Sem compressão de propósito.
    #
    # O arquivo fica um pouco maior, mas carrega muito
    # mais rapidamente durante nossos experimentos.
    # --------------------------------------------------------

    np.savez(
        CACHE_DATASET,

        X=
            np.asarray(
                X
            ),

        y=
            np.asarray(
                y
            ),

        indices_target=
            np.asarray(
                indices_target
            ),

        dezenas=
            np.asarray(
                dezenas
            ),

        matriz_binaria=
            np.asarray(
                matriz_binaria
            )
    )

    metadata = (
        gerar_metadata_esperada(
            caminho_excel=
                caminho_excel,

            caminho_features=
                caminho_features,

            janela_minima=
                janela_minima
        )
    )

    metadata.update({
        "shape_X":
            list(
                X.shape
            ),

        "shape_y":
            list(
                y.shape
            ),

        "shape_indices_target":
            list(
                indices_target.shape
            ),

        "shape_dezenas":
            list(
                dezenas.shape
            ),

        "shape_matriz_binaria":
            list(
                matriz_binaria.shape
            ),

        "total_sorteios":
            int(
                matriz_binaria.shape[0]
            ),

        "quantidade_features":
            int(
                X.shape[1]
            )
            if len(
                X.shape
            ) > 1
            else 0
    })

    with open(
        CACHE_METADATA,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            metadata,
            arquivo,
            indent=2,
            ensure_ascii=False
        )

    tempo = (
        time.time()
        - inicio
    )

    print(
        f"Cache V2 salvo em "
        f"{tempo:.2f}s"
    )

    print(
        f"Dataset: "
        f"{CACHE_DATASET}"
    )

    print(
        f"Metadata: "
        f"{CACHE_METADATA}"
    )


# ============================================================
# CARREGAR CACHE
# ============================================================

def carregar_cache():
    """
    Carrega o dataset previamente calculado.
    """

    if not CACHE_DATASET.exists():
        raise FileNotFoundError(
            f"Cache não encontrado: "
            f"{CACHE_DATASET}"
        )

    print()
    print(
        "Carregando dataset V2 do cache..."
    )

    inicio = time.time()

    with np.load(
        CACHE_DATASET,
        allow_pickle=False
    ) as dados:

        # ----------------------------------------------------
        # Fazemos copy() para não deixar os arrays
        # dependentes do objeto np.load após fechar o arquivo.
        # ----------------------------------------------------

        X = (
            dados[
                "X"
            ]
            .copy()
        )

        y = (
            dados[
                "y"
            ]
            .copy()
        )

        indices_target = (
            dados[
                "indices_target"
            ]
            .copy()
        )

        dezenas = (
            dados[
                "dezenas"
            ]
            .copy()
        )

        matriz_binaria = (
            dados[
                "matriz_binaria"
            ]
            .copy()
        )

    tempo = (
        time.time()
        - inicio
    )

    print(
        f"Cache V2 carregado em "
        f"{tempo:.2f}s"
    )

    print(
        f"X = {X.shape}"
    )

    print(
        f"y = {y.shape}"
    )

    print(
        f"Concursos = "
        f"{matriz_binaria.shape[0]}"
    )

    return (
        X,
        y,
        indices_target,
        dezenas,
        matriz_binaria
    )


# ============================================================
# OBTER DATASET
# ============================================================

def obter_dataset_v2(
    caminho_excel,
    caminho_features,
    df_bolas,
    classe_gerador,
    janela_minima
):
    """
    Interface principal utilizada pelos backtests.

    Se houver cache válido:

        retorna os arrays do disco.

    Caso contrário:

        instancia GeradorFeaturesV2,
        calcula tudo,
        salva,
        retorna.

    Retorno:

        (
            gerador,
            X,
            y,
            indices_target,
            dezenas,
            matriz_binaria
        )

    Quando vem do cache:

        gerador = None

    pois não precisamos instanciar o gerador pesado.
    """

    caminho_excel = Path(
        caminho_excel
    )

    caminho_features = Path(
        caminho_features
    )

    # ========================================================
    # TENTAR CACHE
    # ========================================================

    if cache_valido(
        caminho_excel=
            caminho_excel,

        caminho_features=
            caminho_features,

        janela_minima=
            janela_minima
    ):

        print()
        print(
            "=" * 70
        )

        print(
            "CACHE V2 ENCONTRADO"
        )

        print(
            "=" * 70
        )

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
    # CACHE INVÁLIDO / INEXISTENTE
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "CACHE V2 NÃO ENCONTRADO "
        "OU DESATUALIZADO"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Gerando dataset V2..."
    )

    inicio = time.time()

    gerador = (
        classe_gerador(
            df_bolas
        )
    )

    (
        X,
        y,
        indices_target,
        dezenas
    ) = (
        gerador
        .construir_dataset(
            janela_minima=
                janela_minima
        )
    )

    matriz_binaria = (
        gerador
        .matriz_binaria
    )

    tempo = (
        time.time()
        - inicio
    )

    print()
    print(
        f"Dataset V2 calculado em "
        f"{tempo:.1f}s"
    )

    print(
        f"X = {X.shape}"
    )

    print(
        f"y = {y.shape}"
    )

    # ========================================================
    # SALVAR
    # ========================================================

    salvar_cache(
        caminho_excel=
            caminho_excel,

        caminho_features=
            caminho_features,

        janela_minima=
            janela_minima,

        X=
            X,

        y=
            y,

        indices_target=
            indices_target,

        dezenas=
            dezenas,

        matriz_binaria=
            matriz_binaria
    )

    return (
        gerador,
        X,
        y,
        indices_target,
        dezenas,
        matriz_binaria
    )


# ============================================================
# LIMPAR CACHE
# ============================================================

def limpar_cache():
    """
    Útil para manutenção manual:

        from cache_dataset import limpar_cache
        limpar_cache()
    """

    arquivos = [
        CACHE_DATASET,
        CACHE_METADATA
    ]

    removeu = False

    for arquivo in arquivos:

        if arquivo.exists():

            arquivo.unlink()

            print(
                f"Removido: {arquivo}"
            )

            removeu = True

    if not removeu:

        print(
            "Nenhum cache encontrado."
        )


# ============================================================
# INFO
# ============================================================

def mostrar_info_cache():
    """
    Mostra metadata atualmente armazenada.
    """

    metadata = (
        carregar_metadata()
    )

    print()
    print(
        "=" * 70
    )

    print(
        "INFORMAÇÕES DO CACHE V2"
    )

    print(
        "=" * 70
    )

    if metadata is None:

        print(
            "Cache inexistente."
        )

        return

    for chave, valor in (
        metadata.items()
    ):

        print(
            f"{chave}: {valor}"
        )