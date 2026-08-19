import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
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
# ============================================================

from dados import carregar_resultados

from features_v2_reference import (
    GeradorFeaturesV2,
)

from cache_dataset import (
    obter_dataset_v2,
)

from features_v5 import (
    calcular_features_v5_concurso,
)

from ranking_v4 import (
    criar_ranking_v4,
    treinar_v4,
)

from ranking_v5 import (
    FEATURES_META_V5,
    construir_matriz_v5,
    treinar_v5,
    criar_ranking_v5,
    obter_pesos_v5,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

JANELA_MINIMA = 200

CONCURSOS_META_TREINO = 400

CONCURSOS_TESTE_FINAL = 100


# ============================================================
# V2
# ============================================================

N_ESTIMATORS = 100

MAX_DEPTH = 8

MIN_SAMPLES_LEAF = 10

SEED = 42


# ============================================================
# EXCLUSÕES TESTADAS
# ============================================================

CENARIOS_EXCLUSOES = [
    4,
    5,
    6,
    7,
]


# ============================================================
# SAÍDA
# ============================================================

ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "resultado_backtest_exclusoes_v5.xlsx"
)


# ============================================================
# V2 RANDOM FOREST
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
            "balanced_subsample",
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


# ============================================================
# FEATURES V2 POR DEZENA
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
                - prob_sair,
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
# DATASET V2 COM CACHE
# ============================================================

def preparar_dataset():

    print(
        "=" * 100
    )

    print(
        "BACKTEST V5 - "
        "META-RANKING COM FEATURES RELATIVAS "
        "E SOBREVIVÊNCIA"
    )

    print(
        "=" * 100
    )

    caminho_excel = (
        ROOT
        / "lotofacil_resultados.xlsx"
    )

    caminho_features = (
        ROOT
        / "features_v2_reference.py"
    )

    print()
    print(
        "Carregando histórico..."
    )

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

    # ========================================================
    # QUANDO VEM DO CACHE
    # ========================================================

    if gerador is None:

        class GeradorCache:
            pass

        gerador = (
            GeradorCache()
        )

        gerador.total_sorteios = (
            len(
                matriz_binaria
            )
        )

        gerador.matriz_binaria = (
            matriz_binaria
        )

    print()
    print(
        f"X = {X.shape}"
    )

    print(
        f"y = {y.shape}"
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
# GERAR UM CASO V2 WALK-FORWARD
# ============================================================

def gerar_caso_v2(
    indice_alvo,
    df,
    gerador,
    X,
    y,
    indices_target,
    dezenas,
):

    # ========================================================
    # TREINO
    # ========================================================

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

    # ========================================================
    # TESTE
    # ========================================================

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

    if len(
        X_teste
    ) != 25:

        raise ValueError(
            f"Índice {indice_alvo} "
            f"possui {len(X_teste)} linhas."
        )

    # ========================================================
    # TREINAR V2
    # ========================================================

    modelo_v2 = (
        criar_modelo_v2()
    )

    modelo_v2.fit(
        X_treino,
        y_treino,
    )

    # ========================================================
    # RANKING V2
    # ========================================================

    ranking_v2 = (
        criar_ranking_v2(
            modelo=
                modelo_v2,

            X_teste=
                X_teste,

            dezenas_teste=
                dezenas_teste,
        )
    )

    # ========================================================
    # FEATURES V2 INDIVIDUAIS
    # ========================================================

    features_por_dezena = (
        criar_features_por_dezena(
            X_teste=
                X_teste,

            dezenas_teste=
                dezenas_teste,
        )
    )

    # ========================================================
    # FEATURES EXTRAS V5
    #
    # indice_alvo é o concurso que queremos prever.
    #
    # Portanto o último concurso conhecido é:
    #
    # indice_alvo - 1
    # ========================================================

    extras_v5 = (
        calcular_features_v5_concurso(
            matriz_binaria=
                gerador.matriz_binaria,

            indice_estado=
                indice_alvo - 1,
        )
    )

    # ========================================================
    # RESULTADO REAL
    # ========================================================

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
            range(
                1,
                26
            )
        )
        - sorteadas
    )

    # ========================================================
    # CONCURSO
    # ========================================================

    if (
        "Concurso"
        in df.columns
    ):

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
        JANELA_MINIMA + 1,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "DIVISÃO TEMPORAL"
    )

    print(
        "=" * 70
    )

    print(
        f"Meta-treino: "
        f"{inicio_meta} "
        f"até {inicio_teste - 1}"
    )

    print(
        f"Teste final: "
        f"{inicio_teste} "
        f"até {total - 1}"
    )

    casos_meta = []

    casos_teste = []

    quantidade = (
        total
        - inicio_meta
    )

    inicio_execucao = (
        time.time()
    )

    print()
    print(
        f"Gerando "
        f"{quantidade} "
        f"casos walk-forward V2..."
    )

    # ========================================================
    # LOOP
    # ========================================================

    for numero, indice_alvo in enumerate(
        range(
            inicio_meta,
            total
        ),
        start=1,
    ):

        inicio_caso = (
            time.time()
        )

        caso = (
            gerar_caso_v2(
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
                    dezenas,
            )
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
                f" | último="
                f"{time.time() - inicio_caso:.2f}s"
                f" | total="
                f"{time.time() - inicio_execucao:.1f}s"
            )

    print()
    print(
        f"Casos meta: "
        f"{len(casos_meta)}"
    )

    print(
        f"Casos teste: "
        f"{len(casos_teste)}"
    )

    return (
        casos_meta,
        casos_teste,
    )


# ============================================================
# DATASET META V4
# ============================================================

def construir_dataset_meta_v4(
    casos
):
    """
    Vamos treinar também o V4 para comparar
    usando exatamente os mesmos 400 concursos.
    """

    from ranking_v4 import (
        construir_features_meta,
    )

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
                        ],
                )
            )

            X_meta.append(
                linha
            )

            y_meta.append(
                int(
                    dezena
                    in nao_sorteadas
                )
            )

    return (
        np.asarray(
            X_meta,
            dtype=np.float64,
        ),

        np.asarray(
            y_meta,
            dtype=np.int8,
        ),
    )


# ============================================================
# DATASET META V5
# ============================================================

def construir_dataset_meta_v5(
    casos
):

    X_meta = []

    y_meta = []

    for caso in casos:

        X_concurso = (
            construir_matriz_v5(
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
            )
        )

        nao_sorteadas = (
            caso[
                "nao_sorteadas"
            ]
        )

        for indice_dezena in range(
            25
        ):

            dezena = (
                indice_dezena
                + 1
            )

            X_meta.append(
                X_concurso[
                    indice_dezena
                ]
            )

            y_meta.append(
                int(
                    dezena
                    in nao_sorteadas
                )
            )

    return (
        np.asarray(
            X_meta,
            dtype=np.float64,
        ),

        np.asarray(
            y_meta,
            dtype=np.int8,
        ),
    )


# ============================================================
# TESTE FINAL
# ============================================================

def testar_modelos(
    modelo_v4,
    modelo_v5,
    casos_teste,
):

    resultados = []

    y_real_v4 = []
    probs_v4 = []

    y_real_v5 = []
    probs_v5 = []

    # ========================================================
    # LOOP TESTE
    # ========================================================

    for caso in casos_teste:

        ranking_v2 = (
            caso[
                "ranking_v2"
            ]
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
        # RANKING V4
        # ====================================================

        ranking_v4 = (
            criar_ranking_v4(
                modelo=
                    modelo_v4,

                ranking_v2=
                    ranking_v2,

                features_por_dezena=
                    caso[
                        "features_por_dezena"
                    ],
            )
        )

        # ====================================================
        # RANKING V5
        # ====================================================

        ranking_v5 = (
            criar_ranking_v5(
                modelo=
                    modelo_v5,

                ranking_v2=
                    ranking_v2,

                features_por_dezena=
                    caso[
                        "features_por_dezena"
                    ],

                extras_por_dezena=
                    caso[
                        "extras_v5"
                    ],
            )
        )

        # ====================================================
        # MÉTRICAS PROBABILÍSTICAS
        # ====================================================

        mapa_v4 = {
            item["dezena"]:
                item[
                    "prob_nao_sair_v4"
                ]

            for item
            in ranking_v4
        }

        mapa_v5 = {
            item["dezena"]:
                item[
                    "prob_nao_sair_v5"
                ]

            for item
            in ranking_v5
        }

        for dezena in range(
            1,
            26
        ):

            target = int(
                dezena
                in nao_sorteadas
            )

            y_real_v4.append(
                target
            )

            probs_v4.append(
                mapa_v4[
                    dezena
                ]
            )

            y_real_v5.append(
                target
            )

            probs_v5.append(
                mapa_v5[
                    dezena
                ]
            )

        # ====================================================
        # EXCLUSÕES
        # ====================================================

        for qtd in (
            CENARIOS_EXCLUSOES
        ):

            exclusoes_v2 = {
                item[
                    "dezena"
                ]

                for item
                in ranking_v2[
                    :qtd
                ]
            }

            exclusoes_v4 = {
                item[
                    "dezena"
                ]

                for item
                in ranking_v4[
                    :qtd
                ]
            }

            exclusoes_v5 = {
                item[
                    "dezena"
                ]

                for item
                in ranking_v5[
                    :qtd
                ]
            }

            acertos_v2 = len(
                exclusoes_v2
                & nao_sorteadas
            )

            acertos_v4 = len(
                exclusoes_v4
                & nao_sorteadas
            )

            acertos_v5 = len(
                exclusoes_v5
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

                # ================================
                # ACERTOS
                # ================================

                "acertos_v2":
                    acertos_v2,

                "acertos_v4":
                    acertos_v4,

                "acertos_v5":
                    acertos_v5,

                # ================================
                # GANHO
                # ================================

                "ganho_v4_vs_v2":
                    acertos_v4
                    - acertos_v2,

                "ganho_v5_vs_v2":
                    acertos_v5
                    - acertos_v2,

                "ganho_v5_vs_v4":
                    acertos_v5
                    - acertos_v4,

                # ================================
                # EXCLUSÕES
                # ================================

                "exclusoes_v2":
                    dezenas_para_texto(
                        exclusoes_v2
                    ),

                "exclusoes_v4":
                    dezenas_para_texto(
                        exclusoes_v4
                    ),

                "exclusoes_v5":
                    dezenas_para_texto(
                        exclusoes_v5
                    ),

                "sorteadas_reais":
                    dezenas_para_texto(
                        sorteadas
                    ),
            })

    # ========================================================
    # MÉTRICAS
    # ========================================================

    auc_v4 = roc_auc_score(
        y_real_v4,
        probs_v4,
    )

    brier_v4 = brier_score_loss(
        y_real_v4,
        probs_v4,
    )

    auc_v5 = roc_auc_score(
        y_real_v5,
        probs_v5,
    )

    brier_v5 = brier_score_loss(
        y_real_v5,
        probs_v5,
    )

    return (
        pd.DataFrame(
            resultados
        ),

        auc_v4,
        brier_v4,

        auc_v5,
        brier_v5,
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

        media_v5 = (
            dados[
                "acertos_v5"
            ]
            .mean()
        )

        linhas.append({
            "qtd_exclusoes":
                qtd,

            "qtd_candidatas":
                25 - qtd,

            "concursos":
                len(dados),

            "aleatorio":
                aleatorio,

            "media_v2":
                media_v2,

            "media_v4":
                media_v4,

            "media_v5":
                media_v5,

            "ganho_v4_vs_v2":
                media_v4
                - media_v2,

            "ganho_v5_vs_v2":
                media_v5
                - media_v2,

            "ganho_v5_vs_v4":
                media_v5
                - media_v4,

            "ganho_v5_percentual_vs_aleatorio":
                (
                    (
                        media_v5
                        / aleatorio
                    )
                    - 1
                )
                * 100,

            # =================================================
            # V5 VS V2
            # =================================================

            "v5_melhor_v2":
                int(
                    (
                        dados[
                            "ganho_v5_vs_v2"
                        ]
                        > 0
                    )
                    .sum()
                ),

            "v5_pior_v2":
                int(
                    (
                        dados[
                            "ganho_v5_vs_v2"
                        ]
                        < 0
                    )
                    .sum()
                ),

            "v5_igual_v2":
                int(
                    (
                        dados[
                            "ganho_v5_vs_v2"
                        ]
                        == 0
                    )
                    .sum()
                ),

            # =================================================
            # V5 VS V4
            # =================================================

            "v5_melhor_v4":
                int(
                    (
                        dados[
                            "ganho_v5_vs_v4"
                        ]
                        > 0
                    )
                    .sum()
                ),

            "v5_pior_v4":
                int(
                    (
                        dados[
                            "ganho_v5_vs_v4"
                        ]
                        < 0
                    )
                    .sum()
                ),

            "v5_igual_v4":
                int(
                    (
                        dados[
                            "ganho_v5_vs_v4"
                        ]
                        == 0
                    )
                    .sum()
                ),
        })

    return pd.DataFrame(
        linhas
    )


# ============================================================
# PESOS V5
# ============================================================

def criar_dataframe_pesos_v5(
    modelo_v5
):

    return pd.DataFrame(
        obter_pesos_v5(
            modelo_v5
        )
    )


# ============================================================
# MOSTRAR RESULTADO
# ============================================================

def mostrar_resultados(
    resumo,
    pesos_v5,
    auc_v4,
    brier_v4,
    auc_v5,
    brier_v5,
):

    print()
    print(
        "=" * 125
    )

    print(
        "RESULTADO V5"
    )

    print(
        "=" * 125
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
        "=" * 70
    )

    print(
        "MÉTRICAS"
    )

    print(
        "=" * 70
    )

    print(
        f"AUC V4:   "
        f"{auc_v4:.4f}"
    )

    print(
        f"Brier V4: "
        f"{brier_v4:.4f}"
    )

    print()

    print(
        f"AUC V5:   "
        f"{auc_v5:.4f}"
    )

    print(
        f"Brier V5: "
        f"{brier_v5:.4f}"
    )

    # ========================================================
    # PESOS
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "TOP 30 PESOS V5"
    )

    print(
        "=" * 100
    )

    print(
        pesos_v5[
            [
                "feature",
                "peso",
                "direcao",
            ]
        ]
        .head(30)
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# EXPORTAR
# ============================================================

def exportar(
    resultados,
    resumo,
    pesos_v5,
    auc_v4,
    brier_v4,
    auc_v5,
    brier_v5,
):

    metricas = pd.DataFrame([
        {
            "auc_v4":
                auc_v4,

            "brier_v4":
                brier_v4,

            "auc_v5":
                auc_v5,

            "brier_v5":
                brier_v5,

            "features_v5":
                len(
                    FEATURES_META_V5
                ),

            "meta_treino_concursos":
                CONCURSOS_META_TREINO,

            "teste_final_concursos":
                CONCURSOS_TESTE_FINAL,
        }
    ])

    ARQUIVO_SAIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with pd.ExcelWriter(
        ARQUIVO_SAIDA,
        engine="openpyxl",
    ) as writer:

        resumo.to_excel(
            writer,
            sheet_name="Resumo",
            index=False,
        )

        pesos_v5.to_excel(
            writer,
            sheet_name="Pesos_V5",
            index=False,
        )

        metricas.to_excel(
            writer,
            sheet_name="Metricas",
            index=False,
        )

        resultados.to_excel(
            writer,
            sheet_name="Detalhes",
            index=False,
        )

    print()
    print(
        "Arquivo gerado:"
    )

    print(
        ARQUIVO_SAIDA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    inicio_total = (
        time.time()
    )

    # ========================================================
    # CASOS WALK-FORWARD
    # ========================================================

    (
        casos_meta,
        casos_teste,
    ) = preparar_casos()

    # ========================================================
    # V4
    # ========================================================

    print()
    print(
        "Construindo dataset meta V4..."
    )

    (
        X_meta_v4,
        y_meta_v4,
    ) = (
        construir_dataset_meta_v4(
            casos_meta
        )
    )

    print(
        f"X_meta_v4 = "
        f"{X_meta_v4.shape}"
    )

    print()
    print(
        "Treinando V4..."
    )

    modelo_v4 = (
        treinar_v4(
            X_meta_v4,
            y_meta_v4,
        )
    )

    # ========================================================
    # V5
    # ========================================================

    print()
    print(
        "Construindo dataset meta V5..."
    )

    (
        X_meta_v5,
        y_meta_v5,
    ) = (
        construir_dataset_meta_v5(
            casos_meta
        )
    )

    print(
        f"X_meta_v5 = "
        f"{X_meta_v5.shape}"
    )

    print(
        f"Quantidade de "
        f"features V5 = "
        f"{len(FEATURES_META_V5)}"
    )

    print()
    print(
        "Treinando V5..."
    )

    modelo_v5 = (
        treinar_v5(
            X_meta_v5,
            y_meta_v5,
        )
    )

    # ========================================================
    # PESOS
    # ========================================================

    pesos_v5 = (
        criar_dataframe_pesos_v5(
            modelo_v5
        )
    )

    # ========================================================
    # TESTE FINAL
    # ========================================================

    print()
    print(
        "Executando teste final..."
    )

    (
        resultados,
        auc_v4,
        brier_v4,
        auc_v5,
        brier_v5,
    ) = (
        testar_modelos(
            modelo_v4=
                modelo_v4,

            modelo_v5=
                modelo_v5,

            casos_teste=
                casos_teste,
        )
    )

    # ========================================================
    # RESUMO
    # ========================================================

    resumo = (
        gerar_resumo(
            resultados
        )
    )

    mostrar_resultados(
        resumo=
            resumo,

        pesos_v5=
            pesos_v5,

        auc_v4=
            auc_v4,

        brier_v4=
            brier_v4,

        auc_v5=
            auc_v5,

        brier_v5=
            brier_v5,
    )

    exportar(
        resultados=
            resultados,

        resumo=
            resumo,

        pesos_v5=
            pesos_v5,

        auc_v4=
            auc_v4,

        brier_v4=
            brier_v4,

        auc_v5=
            auc_v5,

        brier_v5=
            brier_v5,
    )

    print()
    print(
        f"Tempo total: "
        f"{time.time() - inicio_total:.1f}s"
    )


if __name__ == "__main__":
    main()