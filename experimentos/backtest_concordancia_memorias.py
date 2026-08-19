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


# ============================================================
# CONFIG
# ============================================================

MEMORIA_CURTA = 6
MEMORIA_LONGA = 94

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
    / "resultado_concordancia_memorias.xlsx"
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
    print("BACKTEST - CONCORDÂNCIA ENTRE MEMÓRIA CURTA E LONGA")
    print("=" * 100)

    df, df_bolas = (
        carregar_resultados(
            caminho_excel
        )
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

    total_teste = (
        BLOCOS_TESTE
        * TAMANHO_BLOCO_TESTE
    )

    primeiro_teste = (
        total
        - total_teste
    )

    maior_memoria = max(
        MEMORIA_CURTA,
        MEMORIA_LONGA
    )

    primeiro_indice = (
        primeiro_teste
        - maior_memoria
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
            or numero % 10 == 0
            or numero == len(indices)
        ):

            print(
                f"{numero:03d}/"
                f"{len(indices)}"
                f" | "
                f"{time.time() - inicio:.1f}s"
            )

    return (
        casos,
        total
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
# ANALISAR UM BLOCO
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

    print()
    print("=" * 100)
    print(f"BLOCO {bloco}")
    print("=" * 100)

    print(
        f"Teste: "
        f"{inicio_teste}.."
        f"{fim_teste - 1}"
    )

    modelo_curto = (
        treinar_para_memoria(
            casos,
            inicio_teste,
            MEMORIA_CURTA
        )
    )

    modelo_longo = (
        treinar_para_memoria(
            casos,
            inicio_teste,
            MEMORIA_LONGA
        )
    )

    resultados = []

    detalhe_posicoes = []

    # ========================================================
    # TESTE
    # ========================================================

    for indice in range(
        inicio_teste,
        fim_teste
    ):

        caso = (
            casos[
                indice
            ]
        )

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
                    )
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

            # =================================================
            # ACERTOS
            # =================================================

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

    return (
        resultados,
        detalhe_posicoes
    )


# ============================================================
# EXECUTAR
# ============================================================

def executar():

    (
        casos,
        total
    ) = (
        preparar_casos()
    )

    resultados = []

    detalhes = []

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
        )
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
                "top"
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

            "media_qtd_consenso":
                dados[
                    "qtd_consenso"
                ]
                .mean(),

            "precisao_consenso":
                precisao,

            "lift_vs_40":
                (
                    precisao - 0.40
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

    print()
    print("=" * 120)
    print("CONCORDÂNCIA CURTO x LONGO")
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
    print("CONSENSO POR BLOCO")
    print("=" * 120)

    print(
        resumo_blocos
        .round(4)
        .to_string(
            index=False
        )
    )

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
    print(
        f"Arquivo:"
    )

    print(
        ARQUIVO_SAIDA
    )

    print()
    print(
        f"Tempo total: "
        f"{time.time() - inicio:.1f}s"
    )


if __name__ == "__main__":
    main()