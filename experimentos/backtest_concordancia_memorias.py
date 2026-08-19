import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATH
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT)
    )


# ============================================================
# IMPORTS
# ============================================================

from dados import (
    carregar_resultados,
)

from cache_dataset import (
    obter_dataset_v2,
)

from features_v2_reference import (
    GeradorFeaturesV2,
)

from features_v5 import (
    calcular_features_v5_concurso,
)

from ranking_v5 import (
    construir_matriz_custom,
    treinar_modelo_meta,
    criar_ranking_custom,
)

from config_ablation_v5 import (
    JANELA_MINIMA,
    N_ESTIMATORS,
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
    SEED,

    GRUPO_RANKING_V2,
    GRUPO_FREQUENCIA_MICRO,
    GRUPO_FREQUENCIA_MESO,
    GRUPO_BASELINE_LONGO,
    GRUPO_TENDENCIA_MICRO,
    GRUPO_MUDANCA_REGIME,
    GRUPO_TENDENCIA_LONGA,
)

from cache_casos import (
    carregar_cache_casos,
    salvar_cache_casos,
)


# ============================================================
# CONFIG
# ============================================================

MEMORIA_CURTA = 6
MEMORIA_LONGA = 94
USAR_CACHE_CASOS = True
# ------------------------------------------------------------
# HISTÓRICO COMPLETO
#
# True:
# percorre todo o histórico possível
#
# False:
# volta ao comportamento antigo:
# últimos 4 blocos de 100
# ------------------------------------------------------------

USAR_HISTORICO_COMPLETO = True

# Apenas para dividir o relatório temporal.
# NÃO é tamanho de treino.
TAMANHO_BLOCO_HISTORICO = 500

# Config antiga, mantida para comparação futura.
BLOCOS_TESTE = 4
TAMANHO_BLOCO_TESTE = 100


TOPS_ANALISADOS = [
    4,
    5,
    6,
    7,
    8,
    9,
]


# ============================================================
# FEATURES MULTIESCALA
# ============================================================

FEATURES_MULTIESCALA = (
    GRUPO_RANKING_V2
    + GRUPO_FREQUENCIA_MICRO
    + GRUPO_FREQUENCIA_MESO
    + GRUPO_BASELINE_LONGO
    + GRUPO_TENDENCIA_MICRO
    + GRUPO_MUDANCA_REGIME
    + GRUPO_TENDENCIA_LONGA
)


# ============================================================
# OUTPUT
# ============================================================

ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "resultado_concordancia_memorias_historico.xlsx"
)


# ============================================================
# V2
# ============================================================

def criar_modelo_v2():

    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


# ============================================================
# HELPERS
# ============================================================

def criar_features_por_dezena(
    X_teste,
    dezenas_teste
):

    nomes = (
        GeradorFeaturesV2
        .nomes_features()
    )

    resultado = {}

    for linha, dezena in zip(
        X_teste,
        dezenas_teste
    ):

        resultado[
            int(dezena)
        ] = {
            nome: float(valor)
            for nome, valor
            in zip(
                nomes,
                linha
            )
        }

    return resultado


def criar_ranking_v2(
    modelo,
    X_teste,
    dezenas_teste
):

    probabilidades = (
        modelo.predict_proba(
            X_teste
        )
    )

    indice_classe_1 = int(
        np.where(
            modelo.classes_ == 1
        )[0][0]
    )

    probs_sair = (
        probabilidades[
            :,
            indice_classe_1
        ]
    )

    ranking = []

    for dezena, prob_sair in zip(
        dezenas_teste,
        probs_sair
    ):

        prob_sair = float(
            prob_sair
        )

        ranking.append({
            "dezena":
                int(dezena),

            "prob_sair":
                prob_sair,

            "prob_nao_sair":
                1.0 - prob_sair,
        })

    ranking.sort(
        key=lambda item:
            item["prob_nao_sair"],
        reverse=True
    )

    return ranking


# ============================================================
# DATASET
# ============================================================

def preparar_dataset():

    caminho_excel = (
        ROOT
        / "lotofacil_resultados.xlsx"
    )

    caminho_features = (
        ROOT
        / "features_v2_reference.py"
    )

    print("=" * 100)
    print(
        "BACKTEST - CONCORDÂNCIA ENTRE MEMÓRIA CURTA E LONGA"
    )
    print("=" * 100)

    print()
    print(
        f"Memória curta: {MEMORIA_CURTA}"
    )

    print(
        f"Memória longa: {MEMORIA_LONGA}"
    )

    print(
        f"Histórico completo: "
        f"{USAR_HISTORICO_COMPLETO}"
    )

    print()

    df, df_bolas = (
        carregar_resultados(
            caminho_excel
        )
    )

    print(
        f"Concursos carregados: "
        f"{len(df_bolas)}"
    )

    (
        gerador,
        X,
        y,
        indices_target,
        dezenas,
        matriz_binaria,
    ) = (
        obter_dataset_v2(
            caminho_excel=
                caminho_excel,

            caminho_features=
                caminho_features,

            df_bolas=
                df_bolas,

            classe_gerador=
                GeradorFeaturesV2,

            janela_minima=
                JANELA_MINIMA,
        )
    )

    if gerador is None:

        class GeradorCache:
            pass

        gerador = GeradorCache()

        gerador.total_sorteios = (
            len(
                matriz_binaria
            )
        )

        gerador.matriz_binaria = (
            matriz_binaria
        )

    return (
        df,
        gerador,
        X,
        y,
        indices_target,
        dezenas,
    )


# ============================================================
# CASO WALK-FORWARD
# ============================================================

def gerar_caso(
    indice_alvo,
    df,
    gerador,
    X,
    y,
    indices_target,
    dezenas,
):

    mascara_treino = (
        indices_target
        < indice_alvo
    )

    mascara_teste = (
        indices_target
        == indice_alvo
    )

    X_treino = (
        X[
            mascara_treino
        ]
    )

    y_treino = (
        y[
            mascara_treino
        ]
    )

    X_teste = (
        X[
            mascara_teste
        ]
    )

    dezenas_teste = (
        dezenas[
            mascara_teste
        ]
    )

    if len(X_teste) != 25:

        raise ValueError(
            f"Índice {indice_alvo}: "
            f"esperava 25 dezenas, "
            f"recebi {len(X_teste)}."
        )

    if len(X_treino) == 0:

        raise ValueError(
            f"Índice {indice_alvo}: "
            "não existem dados anteriores "
            "para treinar V2."
        )

    modelo_v2 = (
        criar_modelo_v2()
    )

    modelo_v2.fit(
        X_treino,
        y_treino
    )

    ranking_v2 = (
        criar_ranking_v2(
            modelo_v2,
            X_teste,
            dezenas_teste
        )
    )

    features_por_dezena = (
        criar_features_por_dezena(
            X_teste,
            dezenas_teste
        )
    )

    extras_v5 = (
        calcular_features_v5_concurso(
            matriz_binaria=
                gerador.matriz_binaria,

            indice_estado=
                indice_alvo - 1
        )
    )

    sorteadas = set(
        np.where(
            gerador
            .matriz_binaria[
                indice_alvo
            ]
            == 1
        )[0]
        + 1
    )

    nao_sorteadas = (
        set(
            range(1, 26)
        )
        - sorteadas
    )

    if "Concurso" in df.columns:

        concurso = int(
            df.iloc[
                indice_alvo
            ]["Concurso"]
        )

    else:

        concurso = (
            indice_alvo + 1
        )

    return {
        "indice":
            indice_alvo,

        "concurso":
            concurso,

        "ranking_v2":
            ranking_v2,

        "features_por_dezena":
            features_por_dezena,

        "extras_v5":
            extras_v5,

        "sorteadas":
            sorteadas,

        "nao_sorteadas":
            nao_sorteadas,
    }


# ============================================================
# PREPARAR CASOS
# ============================================================

def obter_dependencias_cache_casos():

    return {
        "excel":
            ROOT
            / "lotofacil_resultados.xlsx",

        "features_v2":
            ROOT
            / "features_v2_reference.py",

        "features_v5":
            ROOT
            / "features_v5.py",

        "config_ablation_v5":
            ROOT
            / "config_ablation_v5.py",
    }


def obter_parametros_cache_casos():

    return {
        "janela_minima":
            int(
                JANELA_MINIMA
            ),

        "n_estimators":
            int(
                N_ESTIMATORS
            ),

        "max_depth":
            (
                None
                if MAX_DEPTH is None
                else int(
                    MAX_DEPTH
                )
            ),

        "min_samples_leaf":
            int(
                MIN_SAMPLES_LEAF
            ),

        "seed":
            int(
                SEED
            ),
    }

def preparar_casos():

    (
        df,
        gerador,
        X,
        y,
        indices_target,
        dezenas,
    ) = preparar_dataset()

    total = (
        gerador.total_sorteios
    )

    # --------------------------------------------------------
    # Primeiro índice realmente disponível no dataset V2.
    #
    # Isso evita tentar gerar caso antes de existirem
    # features V2 suficientes.
    # --------------------------------------------------------

    targets_disponiveis = np.sort(
        np.unique(
            indices_target
        )
    )

    if len(targets_disponiveis) < 2:
        raise ValueError(
            "Dataset V2 não possui targets suficientes "
            "para iniciar o walk-forward."
        )

    # O primeiro target não pode ser usado como caso,
    # porque não existe nenhum target V2 anterior para
    # treinar o modelo V2.
    #
    # Portanto o primeiro caso walk-forward possível
    # é o SEGUNDO target disponível.
    primeiro_target_v2 = int(
        targets_disponiveis[0]
    )

    primeiro_caso_v2 = int(
        targets_disponiveis[1]
    )

    maior_memoria = max(
        MEMORIA_CURTA,
        MEMORIA_LONGA
    )

    if USAR_HISTORICO_COMPLETO:

        # Precisamos gerar 94 casos anteriores para
        # treinar o meta-modelo longo.
        primeiro_indice = (
            primeiro_caso_v2
        )

        primeiro_teste = (
            primeiro_indice
            + maior_memoria
        )

    else:

        total_teste = (
            BLOCOS_TESTE
            * TAMANHO_BLOCO_TESTE
        )

        primeiro_teste = (
            total
            - total_teste
        )

        primeiro_indice = (
            primeiro_teste
            - maior_memoria
        )

        primeiro_indice = max(
            primeiro_indice,
            primeiro_target_v2
        )

    print()
    print("=" * 100)
    print("PREPARAÇÃO DOS CASOS")
    print("=" * 100)

    print(
        f"Primeiro target V2: "
        f"{primeiro_target_v2}"
    )

    print(
        f"Primeiro índice gerado: "
        f"{primeiro_indice}"
    )
    
    print(
        f"Primeiro caso V2 válido: "
        f"{primeiro_caso_v2}"
    )

    print(
        f"Primeiro índice de teste: "
        f"{primeiro_teste}"
    )

    print(
        f"Último índice: "
        f"{total - 1}"
    )

    # ========================================================
    # CACHE DOS CASOS WALK-FORWARD
    # ========================================================

    arquivos_dependencia = (
        obter_dependencias_cache_casos()
    )

    parametros_cache = (
        obter_parametros_cache_casos()
    )

    if USAR_CACHE_CASOS:

        payload_cache = (
            carregar_cache_casos(
                arquivos_dependencia=
                    arquivos_dependencia,

                parametros=
                    parametros_cache,
            )
        )

        if payload_cache is not None:

            casos = (
                payload_cache[
                    "casos"
                ]
            )

            total_cache = int(
                payload_cache[
                    "total"
                ]
            )

            primeiro_teste_cache = int(
                payload_cache[
                    "primeiro_teste"
                ]
            )

            if (
                total_cache
                == total
                and primeiro_teste_cache
                == primeiro_teste
            ):

                return (
                    casos,
                    total,
                    primeiro_teste,
                )

            print(
                "Cache incompatível com "
                "o intervalo atual."
            )

    casos = {}

    indices = list(
        range(
            primeiro_indice,
            total
        )
    )

    inicio = (
        time.time()
    )

    print()
    print(
        f"Gerando {len(indices)} casos..."
    )

    for numero, indice in enumerate(
        indices,
        start=1
    ):

        casos[
            indice
        ] = (
            gerar_caso(
                indice,
                df,
                gerador,
                X,
                y,
                indices_target,
                dezenas,
            )
        )

        if (
            numero == 1
            or numero % 25 == 0
            or numero == len(indices)
        ):

            print(
                f"{numero:04d}/"
                f"{len(indices)}"
                f" | "
                f"{time.time() - inicio:.1f}s"
            )
            
        if USAR_CACHE_CASOS:

        salvar_cache_casos(
            casos=
                casos,

            total=
                total,

            primeiro_teste=
                primeiro_teste,

            arquivos_dependencia=
                arquivos_dependencia,

            parametros=
                parametros_cache,
        )

    return (
        casos,
        total,
        primeiro_teste,
    )


# ============================================================
# META DATASET
# ============================================================

def construir_dataset_meta(
    casos,
    indices_meta
):

    X_meta = []
    y_meta = []

    for indice in (
        indices_meta
    ):

        caso = (
            casos[
                indice
            ]
        )

        matriz = (
            construir_matriz_custom(
                ranking_v2=
                    caso[
                        "ranking_v2"
                    ],

                features_por_dezena=
                    caso[
                        "features_por_dezena"
                    ],

                extras_por_dezena=
                    caso[
                        "extras_v5"
                    ],

                features_extras=
                    FEATURES_MULTIESCALA,
            )
        )

        for posicao in range(
            25
        ):

            dezena = (
                posicao + 1
            )

            X_meta.append(
                matriz[
                    posicao
                ]
            )

            y_meta.append(
                int(
                    dezena
                    in caso[
                        "nao_sorteadas"
                    ]
                )
            )

    return (
        np.asarray(
            X_meta,
            dtype=np.float64
        ),

        np.asarray(
            y_meta,
            dtype=np.int8
        )
    )


# ============================================================
# TREINAR POR MEMÓRIA
# ============================================================

def treinar_para_memoria(
    casos,
    inicio_teste,
    tamanho
):

    inicio_meta = (
        inicio_teste
        - tamanho
    )

    indices_meta = list(
        range(
            inicio_meta,
            inicio_teste
        )
    )

    faltantes = [
        indice
        for indice
        in indices_meta
        if indice not in casos
    ]

    if faltantes:

        raise ValueError(
            "Casos ausentes para treino: "
            f"{faltantes[:10]}"
        )

    (
        X_meta,
        y_meta
    ) = (
        construir_dataset_meta(
            casos,
            indices_meta
        )
    )

    return (
        treinar_modelo_meta(
            X_meta,
            y_meta
        )
    )


# ============================================================
# ANALISAR INTERVALO
# ============================================================

def analisar_intervalo(
    bloco,
    inicio_teste,
    fim_teste,
    casos
):

    print()
    print("=" * 100)
    print(
        f"BLOCO {bloco}"
    )
    print("=" * 100)

    print(
        f"Teste: "
        f"{inicio_teste}.."
        f"{fim_teste - 1}"
    )

    print(
        "Modo: ROLLING POR CONCURSO"
    )

    resultados = []
    detalhe_posicoes = []

    total_concursos = (
        fim_teste
        - inicio_teste
    )

    inicio_bloco = (
        time.time()
    )

    for numero, indice in enumerate(
        range(
            inicio_teste,
            fim_teste
        ),
        start=1
    ):

        caso = (
            casos[
                indice
            ]
        )

        # ====================================================
        # ROLLING REAL
        #
        # Para prever índice:
        #
        # curto:
        # indice-6 .. indice-1
        #
        # longo:
        # indice-94 .. indice-1
        #
        # Portanto os modelos são treinados NOVAMENTE
        # para CADA concurso-alvo.
        # ====================================================

        modelo_curto = (
            treinar_para_memoria(
                casos=
                    casos,

                inicio_teste=
                    indice,

                tamanho=
                    MEMORIA_CURTA
            )
        )

        modelo_longo = (
            treinar_para_memoria(
                casos=
                    casos,

                inicio_teste=
                    indice,

                tamanho=
                    MEMORIA_LONGA
            )
        )

        # ====================================================
        # RANKING CURTO
        # ====================================================

        ranking_curto = (
            criar_ranking_custom(
                modelo=
                    modelo_curto,

                ranking_v2=
                    caso[
                        "ranking_v2"
                    ],

                features_por_dezena=
                    caso[
                        "features_por_dezena"
                    ],

                extras_por_dezena=
                    caso[
                        "extras_v5"
                    ],

                features_extras=
                    FEATURES_MULTIESCALA,
            )
        )

        # ====================================================
        # RANKING LONGO
        # ====================================================

        ranking_longo = (
            criar_ranking_custom(
                modelo=
                    modelo_longo,

                ranking_v2=
                    caso[
                        "ranking_v2"
                    ],

                features_por_dezena=
                    caso[
                        "features_por_dezena"
                    ],

                extras_por_dezena=
                    caso[
                        "extras_v5"
                    ],

                features_extras=
                    FEATURES_MULTIESCALA,
            )
        )

        # ====================================================
        # POSIÇÕES
        # ====================================================

        pos_curto = {
            item["dezena"]:
                posicao

            for posicao, item
            in enumerate(
                ranking_curto,
                start=1
            )
        }

        pos_longo = {
            item["dezena"]:
                posicao

            for posicao, item
            in enumerate(
                ranking_longo,
                start=1
            )
        }

        # ====================================================
        # DETALHE DAS 25 DEZENAS
        # ====================================================

        for dezena in range(
            1,
            26
        ):

            detalhe_posicoes.append({
                "bloco":
                    bloco,

                "concurso":
                    caso[
                        "concurso"
                    ],

                "indice":
                    indice,

                "dezena":
                    dezena,

                "rank_curto":
                    pos_curto[
                        dezena
                    ],

                "rank_longo":
                    pos_longo[
                        dezena
                    ],

                "nao_saiu":
                    int(
                        dezena
                        in caso[
                            "nao_sorteadas"
                        ]
                    ),
            })

        # ====================================================
        # TOPS
        # ====================================================

        for top in (
            TOPS_ANALISADOS
        ):

            curto = {
                item["dezena"]
                for item
                in ranking_curto[
                    :top
                ]
            }

            longo = {
                item["dezena"]
                for item
                in ranking_longo[
                    :top
                ]
            }

            consenso = (
                curto
                & longo
            )

            somente_curto = (
                curto
                - longo
            )

            somente_longo = (
                longo
                - curto
            )

            acertos_curto = len(
                curto
                & caso[
                    "nao_sorteadas"
                ]
            )

            acertos_longo = len(
                longo
                & caso[
                    "nao_sorteadas"
                ]
            )

            acertos_consenso = len(
                consenso
                & caso[
                    "nao_sorteadas"
                ]
            )

            acertos_somente_curto = len(
                somente_curto
                & caso[
                    "nao_sorteadas"
                ]
            )

            acertos_somente_longo = len(
                somente_longo
                & caso[
                    "nao_sorteadas"
                ]
            )

            resultados.append({
                "bloco":
                    bloco,

                "concurso":
                    caso[
                        "concurso"
                    ],

                "indice":
                    indice,

                "top":
                    top,

                "qtd_consenso":
                    len(
                        consenso
                    ),

                "qtd_somente_curto":
                    len(
                        somente_curto
                    ),

                "qtd_somente_longo":
                    len(
                        somente_longo
                    ),

                "acertos_curto":
                    acertos_curto,

                "acertos_longo":
                    acertos_longo,

                "acertos_consenso":
                    acertos_consenso,

                "acertos_somente_curto":
                    acertos_somente_curto,

                "acertos_somente_longo":
                    acertos_somente_longo,
            })

        # ====================================================
        # PROGRESSO
        # ====================================================

        if (
            numero == 1
            or numero % 25 == 0
            or numero == total_concursos
        ):

            print(
                f"{numero:04d}/"
                f"{total_concursos}"
                f" | índice={indice}"
                f" | curto="
                f"{indice - MEMORIA_CURTA}.."
                f"{indice - 1}"
                f" | longo="
                f"{indice - MEMORIA_LONGA}.."
                f"{indice - 1}"
                f" | tempo="
                f"{time.time() - inicio_bloco:.1f}s"
            )

    return (
        resultados,
        detalhe_posicoes
    )

# ============================================================
# ANALISAR BLOCO ANTIGO
# ============================================================

def analisar_bloco(
    bloco,
    casos,
    total
):

    deslocamento = (
        (
            BLOCOS_TESTE
            - bloco
        )
        * TAMANHO_BLOCO_TESTE
    )

    fim_teste = (
        total
        - deslocamento
    )

    inicio_teste = (
        fim_teste
        - TAMANHO_BLOCO_TESTE
    )

    return (
        analisar_intervalo(
            bloco=
                bloco,

            inicio_teste=
                inicio_teste,

            fim_teste=
                fim_teste,

            casos=
                casos,
        )
    )


# ============================================================
# EXECUTAR
# ============================================================

def executar():

    (
        casos,
        total,
        primeiro_teste,
    ) = (
        preparar_casos()
    )

    resultados = []
    detalhes = []

    if USAR_HISTORICO_COMPLETO:

        blocos = []

        inicio = (
            primeiro_teste
        )

        bloco = 1

        while inicio < total:

            fim = min(
                inicio
                + TAMANHO_BLOCO_HISTORICO,
                total
            )

            blocos.append({
                "bloco":
                    bloco,

                "inicio":
                    inicio,

                "fim":
                    fim,
            })

            inicio = (
                fim
            )

            bloco += 1

        print()
        print("=" * 100)
        print("BLOCOS DO HISTÓRICO")
        print("=" * 100)

        for item in blocos:

            print(
                f"Bloco "
                f"{item['bloco']}: "
                f"{item['inicio']}.."
                f"{item['fim'] - 1}"
            )

        for item in blocos:

            (
                resultado_bloco,
                detalhe_bloco
            ) = (
                analisar_intervalo(
                    bloco=
                        item[
                            "bloco"
                        ],

                    inicio_teste=
                        item[
                            "inicio"
                        ],

                    fim_teste=
                        item[
                            "fim"
                        ],

                    casos=
                        casos,
                )
            )

            resultados.extend(
                resultado_bloco
            )

            detalhes.extend(
                detalhe_bloco
            )

    else:

        for bloco in range(
            1,
            BLOCOS_TESTE + 1
        ):

            (
                resultado_bloco,
                detalhe_bloco
            ) = (
                analisar_bloco(
                    bloco,
                    casos,
                    total
                )
            )

            resultados.extend(
                resultado_bloco
            )

            detalhes.extend(
                detalhe_bloco
            )

    return (
        pd.DataFrame(
            resultados
        ),

        pd.DataFrame(
            detalhes
        ),
    )


# ============================================================
# RESUMO CONSENSO
# ============================================================

def gerar_resumo(
    resultados
):

    linhas = []

    for top in (
        TOPS_ANALISADOS
    ):

        dados = (
            resultados[
                resultados[
                    "top"
                ]
                == top
            ]
        )

        total_consenso = (
            dados[
                "qtd_consenso"
            ]
            .sum()
        )

        total_somente_curto = (
            dados[
                "qtd_somente_curto"
            ]
            .sum()
        )

        total_somente_longo = (
            dados[
                "qtd_somente_longo"
            ]
            .sum()
        )

        acertos_consenso = (
            dados[
                "acertos_consenso"
            ]
            .sum()
        )

        acertos_somente_curto = (
            dados[
                "acertos_somente_curto"
            ]
            .sum()
        )

        acertos_somente_longo = (
            dados[
                "acertos_somente_longo"
            ]
            .sum()
        )

        precisao_consenso = (
            acertos_consenso
            / total_consenso
            if total_consenso > 0
            else np.nan
        )

        precisao_curto = (
            acertos_somente_curto
            / total_somente_curto
            if total_somente_curto > 0
            else np.nan
        )

        precisao_longo = (
            acertos_somente_longo
            / total_somente_longo
            if total_somente_longo > 0
            else np.nan
        )

        linhas.append({
            "top":
                top,

            "media_qtd_consenso":
                dados[
                    "qtd_consenso"
                ]
                .mean(),

            "precisao_consenso":
                precisao_consenso,

            "lift_consenso_vs_40":
                (
                    precisao_consenso
                    - 0.40
                )
                if not np.isnan(
                    precisao_consenso
                )
                else np.nan,

            "precisao_somente_curto":
                precisao_curto,

            "lift_curto_vs_40":
                (
                    precisao_curto
                    - 0.40
                )
                if not np.isnan(
                    precisao_curto
                )
                else np.nan,

            "precisao_somente_longo":
                precisao_longo,

            "lift_longo_vs_40":
                (
                    precisao_longo
                    - 0.40
                )
                if not np.isnan(
                    precisao_longo
                )
                else np.nan,
        })

    return (
        pd.DataFrame(
            linhas
        )
    )


# ============================================================
# RESUMO POR BLOCO
# ============================================================

def gerar_resumo_blocos(
    resultados
):

    linhas = []

    for (
        bloco,
        top
    ), dados in (
        resultados
        .groupby(
            [
                "bloco",
                "top",
            ]
        )
    ):

        qtd_consenso = (
            dados[
                "qtd_consenso"
            ]
            .sum()
        )

        acertos_consenso = (
            dados[
                "acertos_consenso"
            ]
            .sum()
        )

        precisao = (
            acertos_consenso
            / qtd_consenso
            if qtd_consenso > 0
            else np.nan
        )

        linhas.append({
            "bloco":
                bloco,

            "top":
                top,

            "concursos":
                len(
                    dados
                ),

            "media_qtd_consenso":
                dados[
                    "qtd_consenso"
                ]
                .mean(),

            "precisao_consenso":
                precisao,

            "lift_vs_40":
                (
                    precisao
                    - 0.40
                )
                if not np.isnan(
                    precisao
                )
                else np.nan,
        })

    return (
        pd.DataFrame(
            linhas
        )
    )


# ============================================================
# FORÇA POR POSIÇÃO CURTO x LONGO
# ============================================================

def gerar_forca_posicoes(
    detalhes,
    top_maximo=6
):

    dados = (
        detalhes[
            (
                detalhes[
                    "rank_curto"
                ]
                <= top_maximo
            )
            &
            (
                detalhes[
                    "rank_longo"
                ]
                <= top_maximo
            )
        ]
        .copy()
    )

    resumo = (
        dados
        .groupby(
            [
                "rank_curto",
                "rank_longo",
            ],
            as_index=False
        )
        .agg(
            ocorrencias=(
                "nao_saiu",
                "count"
            ),

            acertos=(
                "nao_saiu",
                "sum"
            ),

            precisao=(
                "nao_saiu",
                "mean"
            ),
        )
    )

    resumo[
        "lift_vs_40"
    ] = (
        resumo[
            "precisao"
        ]
        - 0.40
    )

    resumo[
        "lift_percentual_vs_40"
    ] = (
        (
            resumo[
                "precisao"
            ]
            / 0.40
        )
        - 1
    ) * 100

    resumo[
        "soma_ranks"
    ] = (
        resumo[
            "rank_curto"
        ]
        + resumo[
            "rank_longo"
        ]
    )

    resumo[
        "maior_rank"
    ] = (
        resumo[
            [
                "rank_curto",
                "rank_longo",
            ]
        ]
        .max(
            axis=1
        )
    )

    return (
        resumo
        .sort_values(
            [
                "precisao",
                "ocorrencias",
            ],
            ascending=[
                False,
                False,
            ]
        )
    )


# ============================================================
# FORÇA POR POSIÇÃO E BLOCO
# ============================================================

def gerar_forca_posicoes_blocos(
    detalhes,
    top_maximo=6
):

    dados = (
        detalhes[
            (
                detalhes[
                    "rank_curto"
                ]
                <= top_maximo
            )
            &
            (
                detalhes[
                    "rank_longo"
                ]
                <= top_maximo
            )
        ]
        .copy()
    )

    resumo = (
        dados
        .groupby(
            [
                "bloco",
                "rank_curto",
                "rank_longo",
            ],
            as_index=False
        )
        .agg(
            ocorrencias=(
                "nao_saiu",
                "count"
            ),

            acertos=(
                "nao_saiu",
                "sum"
            ),

            precisao=(
                "nao_saiu",
                "mean"
            ),
        )
    )

    resumo[
        "lift_vs_40"
    ] = (
        resumo[
            "precisao"
        ]
        - 0.40
    )

    return resumo


# ============================================================
# CONSENSO POR QUANTIDADE
# ============================================================

def gerar_consenso_por_quantidade(
    resultados,
    top=4
):

    dados = (
        resultados[
            resultados[
                "top"
            ]
            == top
        ]
        .copy()
    )

    linhas = []

    for qtd_consenso, grupo in (
        dados.groupby(
            "qtd_consenso"
        )
    ):

        concursos = (
            len(
                grupo
            )
        )

        total_dezenas = (
            grupo[
                "qtd_consenso"
            ]
            .sum()
        )

        acertos = (
            grupo[
                "acertos_consenso"
            ]
            .sum()
        )

        precisao = (
            acertos
            / total_dezenas
            if total_dezenas > 0
            else np.nan
        )

        linhas.append({
            "top":
                top,

            "qtd_consenso":
                int(
                    qtd_consenso
                ),

            "concursos":
                concursos,

            "percentual_concursos":
                (
                    concursos
                    / len(
                        dados
                    )
                ),

            "total_dezenas_consenso":
                int(
                    total_dezenas
                ),

            "acertos":
                int(
                    acertos
                ),

            "precisao":
                precisao,

            "lift_vs_40":
                (
                    precisao
                    - 0.40
                    if not np.isnan(
                        precisao
                    )
                    else np.nan
                ),

            "lift_percentual_vs_40":
                (
                    (
                        precisao
                        / 0.40
                        - 1
                    )
                    * 100
                    if not np.isnan(
                        precisao
                    )
                    else np.nan
                ),
        })

    return (
        pd.DataFrame(
            linhas
        )
        .sort_values(
            "qtd_consenso"
        )
    )


# ============================================================
# CONSENSO POR QUANTIDADE E BLOCO
# ============================================================

def gerar_consenso_quantidade_blocos(
    resultados,
    top=4
):

    dados = (
        resultados[
            resultados[
                "top"
            ]
            == top
        ]
        .copy()
    )

    linhas = []

    for (
        bloco,
        qtd_consenso
    ), grupo in (
        dados.groupby(
            [
                "bloco",
                "qtd_consenso",
            ]
        )
    ):

        total_dezenas = (
            grupo[
                "qtd_consenso"
            ]
            .sum()
        )

        acertos = (
            grupo[
                "acertos_consenso"
            ]
            .sum()
        )

        precisao = (
            acertos
            / total_dezenas
            if total_dezenas > 0
            else np.nan
        )

        linhas.append({
            "bloco":
                bloco,

            "top":
                top,

            "qtd_consenso":
                int(
                    qtd_consenso
                ),

            "concursos":
                len(
                    grupo
                ),

            "total_dezenas_consenso":
                int(
                    total_dezenas
                ),

            "acertos":
                int(
                    acertos
                ),

            "precisao":
                precisao,

            "lift_vs_40":
                (
                    precisao
                    - 0.40
                    if not np.isnan(
                        precisao
                    )
                    else np.nan
                ),
        })

    return (
        pd.DataFrame(
            linhas
        )
    )


# ============================================================
# FORÇA POR SOMA DOS RANKS
# ============================================================

def gerar_forca_soma_ranks(
    detalhes,
    top_maximo=6
):

    dados = (
        detalhes[
            (
                detalhes[
                    "rank_curto"
                ]
                <= top_maximo
            )
            &
            (
                detalhes[
                    "rank_longo"
                ]
                <= top_maximo
            )
        ]
        .copy()
    )

    dados[
        "soma_ranks"
    ] = (
        dados[
            "rank_curto"
        ]
        + dados[
            "rank_longo"
        ]
    )

    resumo = (
        dados
        .groupby(
            "soma_ranks",
            as_index=False
        )
        .agg(
            ocorrencias=(
                "nao_saiu",
                "count"
            ),

            acertos=(
                "nao_saiu",
                "sum"
            ),

            precisao=(
                "nao_saiu",
                "mean"
            ),
        )
    )

    resumo[
        "lift_vs_40"
    ] = (
        resumo[
            "precisao"
        ]
        - 0.40
    )

    resumo[
        "lift_percentual_vs_40"
    ] = (
        (
            resumo[
                "precisao"
            ]
            / 0.40
        )
        - 1
    ) * 100

    return resumo


# ============================================================
# MAIN
# ============================================================

def main():

    inicio = (
        time.time()
    )

    (
        resultados,
        detalhes
    ) = (
        executar()
    )

    resumo = (
        gerar_resumo(
            resultados
        )
    )

    resumo_blocos = (
        gerar_resumo_blocos(
            resultados
        )
    )

    forca_posicoes = (
        gerar_forca_posicoes(
            detalhes,
            top_maximo=6
        )
    )

    forca_posicoes_blocos = (
        gerar_forca_posicoes_blocos(
            detalhes,
            top_maximo=6
        )
    )

    consenso_qtd_top4 = (
        gerar_consenso_por_quantidade(
            resultados,
            top=4
        )
    )

    consenso_qtd_top5 = (
        gerar_consenso_por_quantidade(
            resultados,
            top=5
        )
    )

    consenso_qtd_top6 = (
        gerar_consenso_por_quantidade(
            resultados,
            top=6
        )
    )

    consenso_qtd_blocos = (
        gerar_consenso_quantidade_blocos(
            resultados,
            top=4
        )
    )

    forca_soma_ranks = (
        gerar_forca_soma_ranks(
            detalhes,
            top_maximo=6
        )
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 120)
    print(
        "CONCORDÂNCIA CURTO x LONGO"
    )
    print("=" * 120)

    print(
        resumo
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print(
        "CONSENSO POR BLOCO"
    )
    print("=" * 120)

    print(
        resumo_blocos
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print(
        "FORÇA POR POSIÇÃO CURTO x LONGO"
    )
    print("=" * 120)

    print(
        forca_posicoes
        .head(40)
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print(
        "CONSENSO POR QUANTIDADE - TOP 4"
    )
    print("=" * 120)

    print(
        consenso_qtd_top4
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print(
        "FORÇA POR SOMA DOS RANKS"
    )
    print("=" * 120)

    print(
        forca_soma_ranks
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # EXCEL
    # ========================================================

    with pd.ExcelWriter(
        ARQUIVO_SAIDA,
        engine="openpyxl"
    ) as writer:

        resumo.to_excel(
            writer,
            sheet_name="Resumo",
            index=False
        )

        resumo_blocos.to_excel(
            writer,
            sheet_name="Blocos",
            index=False
        )

        forca_posicoes.to_excel(
            writer,
            sheet_name="Forca_Posicoes",
            index=False
        )

        forca_posicoes_blocos.to_excel(
            writer,
            sheet_name="Forca_Pos_Blocos",
            index=False
        )

        consenso_qtd_top4.to_excel(
            writer,
            sheet_name="Qtd_Consenso_Top4",
            index=False
        )

        consenso_qtd_top5.to_excel(
            writer,
            sheet_name="Qtd_Consenso_Top5",
            index=False
        )

        consenso_qtd_top6.to_excel(
            writer,
            sheet_name="Qtd_Consenso_Top6",
            index=False
        )

        consenso_qtd_blocos.to_excel(
            writer,
            sheet_name="Qtd_Cons_Blocos",
            index=False
        )

        forca_soma_ranks.to_excel(
            writer,
            sheet_name="Forca_Soma_Ranks",
            index=False
        )

        resultados.to_excel(
            writer,
            sheet_name="Concursos",
            index=False
        )

        detalhes.to_excel(
            writer,
            sheet_name="Posicoes",
            index=False
        )

    print()
    print("=" * 120)

    print(
        f"Arquivo:"
    )

    print(
        ARQUIVO_SAIDA
    )

    print()

    print(
        f"Concursos analisados: "
        f"{resultados['concurso'].nunique()}"
    )

    print(
        f"Registros de posições: "
        f"{len(detalhes)}"
    )

    print(
        f"Tempo total: "
        f"{time.time() - inicio:.1f}s"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()