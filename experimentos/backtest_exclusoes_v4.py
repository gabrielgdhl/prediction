import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss
)


# ============================================================
# PATH DO PROJETO
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT)
    )


# ============================================================
# IMPORTS DO PROJETO
#
# IMPORTANTE:
# precisam vir DEPOIS de adicionar ROOT ao sys.path
# ============================================================

from dados import carregar_resultados

from features_v2_reference import (
    GeradorFeaturesV2
)

from ranking_v4 import (
    FEATURES_META,
    construir_features_meta,
    treinar_v4,
    criar_ranking_v4,
    obter_pesos_modelo
)

from cache_dataset import (
    obter_dataset_v2
)


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT)
    )


from dados import carregar_resultados

from features_v2_reference import (
    GeradorFeaturesV2
)

from ranking_v4 import (
    FEATURES_META,
    construir_features_meta,
    treinar_v4,
    criar_ranking_v4,
    obter_pesos_modelo
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

JANELA_MINIMA = 200


# ------------------------------------------------------------
# META-MODELO
#
# Esses concursos são usados para a Logistic Regression
# aprender os pesos.
# ------------------------------------------------------------

CONCURSOS_META_TREINO = 400


# ------------------------------------------------------------
# TESTE FINAL
#
# O V4 nunca aprende os pesos olhando esses concursos.
# ------------------------------------------------------------

CONCURSOS_TESTE_FINAL = 100


# ------------------------------------------------------------
# V2
#
# Mantemos EXATAMENTE os parâmetros já utilizados
# no baseline V2.
# ------------------------------------------------------------

N_ESTIMATORS = 100

MAX_DEPTH = 8

MIN_SAMPLES_LEAF = 10

SEED = 42


CENARIOS_EXCLUSOES = [
    4,
    5,
    6,
    7
]


ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "resultado_backtest_exclusoes_v4.xlsx"
)


# ============================================================
# V2
# ============================================================

def criar_modelo_v2():

    return RandomForestClassifier(
        n_estimators=
            N_ESTIMATORS,

        max_depth=
            MAX_DEPTH,

        min_samples_leaf=
            MIN_SAMPLES_LEAF,

        random_state=
            SEED,

        n_jobs=-1,

        class_weight=
            "balanced_subsample"
    )


# ============================================================
# HELPERS
# ============================================================

def dezenas_para_texto(
    dezenas
):

    return " ".join(
        f"{d:02d}"
        for d
        in sorted(
            dezenas
        )
    )


def esperado_aleatorio(
    quantidade
):

    return (
        quantidade
        * (10 / 25)
    )


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
            nome:
                float(valor)

            for nome, valor
            in zip(
                nomes,
                linha
            )
        }

    return resultado


# ============================================================
# RANKING V2
# ============================================================

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

    classes = (
        modelo.classes_
    )

    indice_classe_1 = (
        np.where(
            classes == 1
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
                1.0
                - prob_sair
        })

    ranking.sort(
        key=lambda item:
            item[
                "prob_nao_sair"
            ],
        reverse=True
    )

    return ranking


# ============================================================
# DATASET BASE
# ============================================================

def preparar_dataset():

    print("=" * 100)
    print("BACKTEST V4 - META-RANKING")
    print("=" * 100)

    caminho_excel = (
        ROOT
        / "lotofacil_resultados.xlsx"
    )

    caminho_features = (
        ROOT
        / "features_v2_reference.py"
    )

    print()
    print("Carregando histórico...")

    df, df_bolas = carregar_resultados(
        caminho_excel
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
        matriz_binaria
    ) = obter_dataset_v2(
        caminho_excel=caminho_excel,
        caminho_features=caminho_features,
        df_bolas=df_bolas,
        classe_gerador=GeradorFeaturesV2,
        janela_minima=JANELA_MINIMA
    )

    if gerador is None:

        class GeradorCache:
            pass

        gerador = GeradorCache()

        gerador.total_sorteios = (
            len(matriz_binaria)
        )

        gerador.matriz_binaria = (
            matriz_binaria
        )

    print()
    print(f"X = {X.shape}")
    print(f"y = {y.shape}")

    return (
        df,
        gerador,
        X,
        y,
        indices_target,
        dezenas
    )


# ============================================================
# GERAR UM CASO WALK-FORWARD
# ============================================================

def gerar_caso_v2(
    indice_alvo,
    df,
    gerador,
    X,
    y,
    indices_target,
    dezenas
):
    """
    Muito importante:

    Para prever indice_alvo:

        V2 só treina com targets < indice_alvo.
    """

    mascara_treino = (
        indices_target
        < indice_alvo
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

    mascara_teste = (
        indices_target
        == indice_alvo
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
            f"Índice {indice_alvo} "
            f"possui {len(X_teste)} linhas."
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

    sorteadas = set(
        np.where(
            gerador
            .matriz_binaria[
                indice_alvo
            ]
            == 1
        )[0] + 1
    )

    nao_sorteadas = (
        set(
            range(
                1,
                26
            )
        )
        - sorteadas
    )

    if "Concurso" in df.columns:

        concurso = int(
            df.iloc[
                indice_alvo
            ][
                "Concurso"
            ]
        )

    else:

        concurso = (
            indice_alvo
            + 1
        )

    return {
        "concurso":
            concurso,

        "indice":
            indice_alvo,

        "ranking_v2":
            ranking_v2,

        "features_por_dezena":
            features_por_dezena,

        "sorteadas":
            sorteadas,

        "nao_sorteadas":
            nao_sorteadas
    }


# ============================================================
# GERAR CASOS
# ============================================================

def preparar_casos():

    (
        df,
        gerador,
        X,
        y,
        indices_target,
        dezenas
    ) = preparar_dataset()

    total = (
        gerador
        .total_sorteios
    )

    inicio_teste = (
        total
        - CONCURSOS_TESTE_FINAL
    )

    inicio_meta = (
        inicio_teste
        - CONCURSOS_META_TREINO
    )

    inicio_meta = max(
        inicio_meta,
        JANELA_MINIMA + 1
    )

    print()
    print(
        "DIVISÃO TEMPORAL"
    )

    print(
        "-" * 60
    )

    print(
        f"Meta-treino:"
        f" índices {inicio_meta} "
        f"até {inicio_teste - 1}"
    )

    print(
        f"Teste final:"
        f" índices {inicio_teste} "
        f"até {total - 1}"
    )

    casos_meta = []
    casos_teste = []

    indices_processar = range(
        inicio_meta,
        total
    )

    quantidade = (
        total
        - inicio_meta
    )

    inicio_execucao = (
        time.time()
    )

    print()
    print(
        f"Gerando {quantidade} "
        f"previsões walk-forward V2..."
    )

    for numero, indice_alvo in enumerate(
        indices_processar,
        start=1
    ):

        caso = gerar_caso_v2(
            indice_alvo=
                indice_alvo,

            df=
                df,

            gerador=
                gerador,

            X=
                X,

            y=
                y,

            indices_target=
                indices_target,

            dezenas=
                dezenas
        )

        if (
            indice_alvo
            < inicio_teste
        ):

            casos_meta.append(
                caso
            )

        else:

            casos_teste.append(
                caso
            )

        if (
            numero == 1
            or numero % 10 == 0
            or numero == quantidade
        ):

            print(
                f"{numero:03d}/"
                f"{quantidade}"
                f" | total="
                f"{time.time() - inicio_execucao:.1f}s"
            )

    print()
    print(
        f"Casos meta-treino: "
        f"{len(casos_meta)}"
    )

    print(
        f"Casos teste final: "
        f"{len(casos_teste)}"
    )

    return (
        casos_meta,
        casos_teste
    )


# ============================================================
# DATASET DO META-MODELO
# ============================================================

def construir_dataset_meta(
    casos
):

    X_meta = []
    y_meta = []

    for caso in casos:

        ranking_map = {
            item["dezena"]:
                item

            for item
            in caso[
                "ranking_v2"
            ]
        }

        nao_sorteadas = (
            caso[
                "nao_sorteadas"
            ]
        )

        for dezena in range(
            1,
            26
        ):

            linha = (
                construir_features_meta(
                    ranking_v2_item=
                        ranking_map[
                            dezena
                        ],

                    features_dezena=
                        caso[
                            "features_por_dezena"
                        ][
                            dezena
                        ]
                )
            )

            target = int(
                dezena
                in nao_sorteadas
            )

            X_meta.append(
                linha
            )

            y_meta.append(
                target
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
# TOP-2 SOBERANO
# ============================================================

def selecionar_v4_com_top2(
    ranking_v2,
    ranking_v4,
    quantidade
):
    """
    Preserva:

        Top 1 e Top 2 do V2

    e completa o restante usando o V4.
    """

    exclusoes = []

    # ========================================================
    # TOP 2 V2
    # ========================================================

    for item in (
        ranking_v2[
            :2
        ]
    ):

        exclusoes.append(
            item[
                "dezena"
            ]
        )

        if (
            len(exclusoes)
            >= quantidade
        ):
            return set(
                exclusoes
            )

    # ========================================================
    # RESTANTE V4
    # ========================================================

    for item in ranking_v4:

        dezena = (
            item[
                "dezena"
            ]
        )

        if dezena in exclusoes:
            continue

        exclusoes.append(
            dezena
        )

        if (
            len(exclusoes)
            >= quantidade
        ):
            break

    return set(
        exclusoes
    )


# ============================================================
# TESTE FINAL
# ============================================================

def testar_modelo(
    modelo_v4,
    casos_teste
):

    resultados = []

    y_reais_probabilidade = []

    probs_v4_todas = []

    for caso in casos_teste:

        ranking_v2 = (
            caso[
                "ranking_v2"
            ]
        )

        ranking_v4 = (
            criar_ranking_v4(
                modelo=
                    modelo_v4,

                ranking_v2=
                    ranking_v2,

                features_por_dezena=
                    caso[
                        "features_por_dezena"
                    ]
            )
        )

        nao_sorteadas = (
            caso[
                "nao_sorteadas"
            ]
        )

        sorteadas = (
            caso[
                "sorteadas"
            ]
        )

        # ====================================================
        # MÉTRICAS DE PROBABILIDADE
        # ====================================================

        mapa_v4 = {
            item["dezena"]:
                item[
                    "prob_nao_sair_v4"
                ]

            for item
            in ranking_v4
        }

        for dezena in range(
            1,
            26
        ):

            y_reais_probabilidade.append(
                int(
                    dezena
                    in nao_sorteadas
                )
            )

            probs_v4_todas.append(
                mapa_v4[
                    dezena
                ]
            )

        # ====================================================
        # CENÁRIOS
        # ====================================================

        for qtd in (
            CENARIOS_EXCLUSOES
        ):

            # ------------------------------------------------
            # V2
            # ------------------------------------------------

            exclusoes_v2 = {
                item[
                    "dezena"
                ]

                for item
                in ranking_v2[
                    :qtd
                ]
            }

            # ------------------------------------------------
            # V4 PURO
            # ------------------------------------------------

            exclusoes_v4 = {
                item[
                    "dezena"
                ]

                for item
                in ranking_v4[
                    :qtd
                ]
            }

            # ------------------------------------------------
            # V4 + TOP2 V2
            # ------------------------------------------------

            exclusoes_v4_top2 = (
                selecionar_v4_com_top2(
                    ranking_v2=
                        ranking_v2,

                    ranking_v4=
                        ranking_v4,

                    quantidade=
                        qtd
                )
            )

            acertos_v2 = len(
                exclusoes_v2
                & nao_sorteadas
            )

            acertos_v4 = len(
                exclusoes_v4
                & nao_sorteadas
            )

            acertos_v4_top2 = len(
                exclusoes_v4_top2
                & nao_sorteadas
            )

            resultados.append({
                "concurso":
                    caso[
                        "concurso"
                    ],

                "qtd_exclusoes":
                    qtd,

                "qtd_candidatas":
                    25 - qtd,

                "acertos_v2":
                    acertos_v2,

                "acertos_v4":
                    acertos_v4,

                "acertos_v4_top2":
                    acertos_v4_top2,

                "ganho_v4_vs_v2":
                    acertos_v4
                    - acertos_v2,

                "ganho_top2_vs_v2":
                    acertos_v4_top2
                    - acertos_v2,

                "exclusoes_v2":
                    dezenas_para_texto(
                        exclusoes_v2
                    ),

                "exclusoes_v4":
                    dezenas_para_texto(
                        exclusoes_v4
                    ),

                "exclusoes_v4_top2":
                    dezenas_para_texto(
                        exclusoes_v4_top2
                    ),

                "sorteadas_reais":
                    dezenas_para_texto(
                        sorteadas
                    )
            })

    # ========================================================
    # AUC / BRIER
    # ========================================================

    auc = roc_auc_score(
        y_reais_probabilidade,
        probs_v4_todas
    )

    brier = brier_score_loss(
        y_reais_probabilidade,
        probs_v4_todas
    )

    return (
        pd.DataFrame(
            resultados
        ),
        auc,
        brier
    )


# ============================================================
# RESUMO
# ============================================================

def gerar_resumo(
    resultados
):

    linhas = []

    for qtd in (
        CENARIOS_EXCLUSOES
    ):

        dados = (
            resultados[
                resultados[
                    "qtd_exclusoes"
                ]
                == qtd
            ]
        )

        aleatorio = (
            esperado_aleatorio(
                qtd
            )
        )

        media_v2 = (
            dados[
                "acertos_v2"
            ]
            .mean()
        )

        media_v4 = (
            dados[
                "acertos_v4"
            ]
            .mean()
        )

        media_top2 = (
            dados[
                "acertos_v4_top2"
            ]
            .mean()
        )

        linhas.append({
            "qtd_exclusoes":
                qtd,

            "qtd_candidatas":
                25 - qtd,

            "concursos":
                len(
                    dados
                ),

            "aleatorio":
                aleatorio,

            "media_v2":
                media_v2,

            "media_v4":
                media_v4,

            "media_v4_top2":
                media_top2,

            "ganho_v4_vs_v2":
                media_v4
                - media_v2,

            "ganho_top2_vs_v2":
                media_top2
                - media_v2,

            "ganho_v4_percentual_vs_aleatorio":
                (
                    (
                        media_v4
                        / aleatorio
                    )
                    - 1
                )
                * 100,

            "ganho_top2_percentual_vs_aleatorio":
                (
                    (
                        media_top2
                        / aleatorio
                    )
                    - 1
                )
                * 100,

            "v4_melhor_v2":
                int(
                    (
                        dados[
                            "ganho_v4_vs_v2"
                        ]
                        > 0
                    )
                    .sum()
                ),

            "v4_pior_v2":
                int(
                    (
                        dados[
                            "ganho_v4_vs_v2"
                        ]
                        < 0
                    )
                    .sum()
                ),

            "top2_melhor_v2":
                int(
                    (
                        dados[
                            "ganho_top2_vs_v2"
                        ]
                        > 0
                    )
                    .sum()
                ),

            "top2_pior_v2":
                int(
                    (
                        dados[
                            "ganho_top2_vs_v2"
                        ]
                        < 0
                    )
                    .sum()
                )
        })

    return pd.DataFrame(
        linhas
    )


# ============================================================
# PESOS
# ============================================================

def criar_dataframe_pesos(
    modelo
):

    pesos = (
        obter_pesos_modelo(
            modelo
        )
    )

    return pd.DataFrame(
        pesos
    )


# ============================================================
# OUTPUT
# ============================================================

def mostrar_resultados(
    resumo,
    pesos,
    auc,
    brier
):

    print()
    print(
        "=" * 120
    )

    print(
        "RESULTADO FINAL V4"
    )

    print(
        "=" * 120
    )

    print(
        resumo
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"AUC V4:   "
        f"{auc:.4f}"
    )

    print(
        f"Brier V4: "
        f"{brier:.4f}"
    )

    print()
    print(
        "=" * 90
    )

    print(
        "PESOS APRENDIDOS PELO META-MODELO"
    )

    print(
        "=" * 90
    )

    print(
        pesos[
            [
                "feature",
                "peso",
                "direcao"
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# EXPORTAÇÃO
# ============================================================

def exportar(
    resultados,
    resumo,
    pesos,
    auc,
    brier
):

    metricas = pd.DataFrame([
        {
            "auc_v4":
                auc,

            "brier_v4":
                brier,

            "meta_features":
                len(
                    FEATURES_META
                ),

            "meta_treino_concursos":
                CONCURSOS_META_TREINO,

            "teste_final_concursos":
                CONCURSOS_TESTE_FINAL
        }
    ])

    with pd.ExcelWriter(
        ARQUIVO_SAIDA,
        engine="openpyxl"
    ) as writer:

        resumo.to_excel(
            writer,
            sheet_name="Resumo",
            index=False
        )

        pesos.to_excel(
            writer,
            sheet_name="Pesos",
            index=False
        )

        metricas.to_excel(
            writer,
            sheet_name="Metricas",
            index=False
        )

        resultados.to_excel(
            writer,
            sheet_name="Detalhes",
            index=False
        )

    print()
    print(
        f"Arquivo gerado:"
    )

    print(
        ARQUIVO_SAIDA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    inicio = (
        time.time()
    )

    # ========================================================
    # 1. GERAR PREVISÕES V2 HISTÓRICAS
    # ========================================================

    (
        casos_meta,
        casos_teste
    ) = preparar_casos()

    # ========================================================
    # 2. DATASET DO META-MODELO
    # ========================================================

    print()
    print(
        "Construindo dataset do "
        "meta-modelo..."
    )

    (
        X_meta,
        y_meta
    ) = (
        construir_dataset_meta(
            casos_meta
        )
    )

    print(
        f"X_meta = "
        f"{X_meta.shape}"
    )

    print(
        f"y_meta = "
        f"{y_meta.shape}"
    )

    # ========================================================
    # 3. TREINAR V4
    # ========================================================

    print()
    print(
        "Treinando Logistic Regression V4..."
    )

    modelo_v4 = (
        treinar_v4(
            X_meta,
            y_meta
        )
    )

    # ========================================================
    # 4. PESOS
    # ========================================================

    pesos = (
        criar_dataframe_pesos(
            modelo_v4
        )
    )

    # ========================================================
    # 5. TESTE FINAL
    # ========================================================

    print()
    print(
        "Executando teste final "
        "nos últimos 100 concursos..."
    )

    (
        resultados,
        auc,
        brier
    ) = testar_modelo(
        modelo_v4,
        casos_teste
    )

    resumo = (
        gerar_resumo(
            resultados
        )
    )

    # ========================================================
    # 6. OUTPUT
    # ========================================================

    mostrar_resultados(
        resumo,
        pesos,
        auc,
        brier
    )

    exportar(
        resultados,
        resumo,
        pesos,
        auc,
        brier
    )

    print()
    print(
        f"Tempo total: "
        f"{time.time() - inicio:.1f}s"
    )


if __name__ == "__main__":
    main()