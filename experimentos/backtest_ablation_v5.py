import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
)


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
# PROJETO
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
    obter_pesos_custom,
    obter_features_modelo,
)

from config_ablation_v5 import (
    JANELA_MINIMA,

    META_TREINO_CONCURSOS,

    BLOCOS_TESTE,
    TAMANHO_BLOCO_TESTE,

    N_ESTIMATORS,
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
    SEED,

    CENARIOS_EXCLUSOES,

    EXPERIMENTOS,
)


# ============================================================
# OUTPUT
# ============================================================

ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "resultado_ablation_v5.xlsx"
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
            "balanced_subsample",
    )


# ============================================================
# HELPERS
# ============================================================

def esperado_aleatorio(
    qtd
):

    return (
        qtd
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
                int(
                    dezena
                ),

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
# DATASET COM CACHE
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

    print(
        "=" * 100
    )

    print(
        "ABLATION ANALYSIS V5"
    )

    print(
        "=" * 100
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
        f"Concursos: "
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

    if len(
        X_teste
    ) != 25:

        raise ValueError(
            f"Índice {indice_alvo}: "
            f"{len(X_teste)} dezenas."
        )

    modelo = (
        criar_modelo_v2()
    )

    modelo.fit(
        X_treino,
        y_treino
    )

    ranking_v2 = (
        criar_ranking_v2(
            modelo,
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
            range(
                1,
                26
            )
        )
        - sorteadas
    )

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
# PREPARAR TODOS OS CASOS NECESSÁRIOS
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

    total_teste = (
        BLOCOS_TESTE
        * TAMANHO_BLOCO_TESTE
    )

    primeiro_teste = (
        total
        - total_teste
    )

    primeiro_meta = (
        primeiro_teste
        - META_TREINO_CONCURSOS
    )

    primeiro_meta = max(
        primeiro_meta,
        JANELA_MINIMA + 1
    )

    print()
    print(
        f"Gerando casos de "
        f"{primeiro_meta} "
        f"até {total - 1}"
    )

    print(
        f"Total: "
        f"{total - primeiro_meta}"
    )

    casos = {}

    inicio = (
        time.time()
    )

    indices = list(
        range(
            primeiro_meta,
            total
        )
    )

    for numero, indice in enumerate(
        indices,
        start=1
    ):

        casos[
            indice
        ] = (
            gerar_caso(
                indice_alvo=
                    indice,

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
            numero == 1
            or numero % 10 == 0
            or numero == len(indices)
        ):

            print(
                f"{numero:03d}/"
                f"{len(indices)}"
                f" | total="
                f"{time.time() - inicio:.1f}s"
            )

    return (
        casos,
        total
    )


# ============================================================
# DATASET META
# ============================================================

def construir_dataset_meta(
    casos,
    indices,
    features_extras
):

    X_meta = []
    y_meta = []

    for indice in indices:

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
                    features_extras
            )
        )

        for posicao in range(
            25
        ):

            dezena = (
                posicao
                + 1
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
# TESTAR EXPERIMENTO
# ============================================================

def testar_experimento(
    nome,
    features_extras,
    modelo,
    casos,
    indices_teste,
    bloco
):

    resultados = []

    y_real = []
    probs = []

    for indice in indices_teste:

        caso = (
            casos[
                indice
            ]
        )

        ranking = (
            criar_ranking_custom(
                modelo=
                    modelo,

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
                    features_extras
            )
        )

        mapa_probs = {
            item["dezena"]:
                item[
                    "prob_nao_sair"
                ]

            for item in ranking
        }

        for dezena in range(
            1,
            26
        ):

            y_real.append(
                int(
                    dezena
                    in caso[
                        "nao_sorteadas"
                    ]
                )
            )

            probs.append(
                mapa_probs[
                    dezena
                ]
            )

        for qtd in (
            CENARIOS_EXCLUSOES
        ):

            exclusoes = {
                item[
                    "dezena"
                ]

                for item
                in ranking[
                    :qtd
                ]
            }

            acertos = len(
                exclusoes
                & caso[
                    "nao_sorteadas"
                ]
            )

            resultados.append({
                "bloco":
                    bloco,

                "modelo":
                    nome,

                "concurso":
                    caso[
                        "concurso"
                    ],

                "qtd_exclusoes":
                    qtd,

                "acertos":
                    acertos,
            })

    auc = (
        roc_auc_score(
            y_real,
            probs
        )
    )

    brier = (
        brier_score_loss(
            y_real,
            probs
        )
    )

    return (
        resultados,
        auc,
        brier
    )


# ============================================================
# EXECUTAR ABLATION
# ============================================================

def executar():

    (
        casos,
        total
    ) = preparar_casos()

    resultados = []

    metricas = []

    pesos = []

    # ========================================================
    # BLOCOS TEMPORAIS
    # ========================================================

    for bloco in range(
        BLOCOS_TESTE
    ):

        deslocamento = (
            (
                BLOCOS_TESTE
                - bloco
                - 1
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

        fim_meta = (
            inicio_teste
        )

        inicio_meta = (
            fim_meta
            - META_TREINO_CONCURSOS
        )

        indices_meta = list(
            range(
                inicio_meta,
                fim_meta
            )
        )

        indices_teste = list(
            range(
                inicio_teste,
                fim_teste
            )
        )

        print()
        print(
            "=" * 100
        )

        print(
            f"BLOCO {bloco + 1}"
        )

        print(
            "=" * 100
        )

        print(
            f"Meta treino: "
            f"{inicio_meta}.."
            f"{fim_meta - 1}"
        )

        print(
            f"Teste: "
            f"{inicio_teste}.."
            f"{fim_teste - 1}"
        )

        # ====================================================
        # EXPERIMENTOS
        # ====================================================

        for nome, features_extras in (
            EXPERIMENTOS.items()
        ):

            print(
                f"  → {nome}"
            )

            features_modelo = (
                obter_features_modelo(
                    features_extras
                )
            )

            (
                X_meta,
                y_meta
            ) = (
                construir_dataset_meta(
                    casos=
                        casos,

                    indices=
                        indices_meta,

                    features_extras=
                        features_extras
                )
            )

            modelo = (
                treinar_modelo_meta(
                    X_meta,
                    y_meta
                )
            )

            (
                resultados_exp,
                auc,
                brier
            ) = (
                testar_experimento(
                    nome=
                        nome,

                    features_extras=
                        features_extras,

                    modelo=
                        modelo,

                    casos=
                        casos,

                    indices_teste=
                        indices_teste,

                    bloco=
                        bloco + 1
                )
            )

            resultados.extend(
                resultados_exp
            )

            metricas.append({
                "bloco":
                    bloco + 1,

                "modelo":
                    nome,

                "qtd_features":
                    len(
                        features_modelo
                    ),

                "auc":
                    auc,

                "brier":
                    brier,
            })

            pesos_modelo = (
                obter_pesos_custom(
                    modelo,
                    features_extras
                )
            )

            for item in (
                pesos_modelo
            ):

                pesos.append({
                    "bloco":
                        bloco + 1,

                    "modelo":
                        nome,

                    **item
                })

    return (
        pd.DataFrame(
            resultados
        ),

        pd.DataFrame(
            metricas
        ),

        pd.DataFrame(
            pesos
        )
    )


# ============================================================
# RESUMO
# ============================================================

def gerar_resumo(
    resultados
):

    resumo = (
        resultados
        .groupby(
            [
                "modelo",
                "qtd_exclusoes"
            ],
            as_index=False
        )
        .agg(
            concursos=(
                "concurso",
                "count"
            ),

            media_acertos=(
                "acertos",
                "mean"
            ),

            desvio_acertos=(
                "acertos",
                "std"
            ),
        )
    )

    resumo[
        "aleatorio"
    ] = (
        resumo[
            "qtd_exclusoes"
        ]
        * (
            10 / 25
        )
    )

    resumo[
        "ganho_absoluto"
    ] = (
        resumo[
            "media_acertos"
        ]
        - resumo[
            "aleatorio"
        ]
    )

    resumo[
        "ganho_percentual"
    ] = (
        (
            resumo[
                "media_acertos"
            ]
            / resumo[
                "aleatorio"
            ]
        )
        - 1
    ) * 100

    # ========================================================
    # SCORE AGREGADO
    # ========================================================

    scores = []

    for modelo in (
        resumo[
            "modelo"
        ]
        .unique()
    ):

        dados = (
            resumo[
                resumo[
                    "modelo"
                ]
                == modelo
            ]
        )

        score = float(
            np.mean(
                dados[
                    "media_acertos"
                ]
                / dados[
                    "aleatorio"
                ]
            )
        )

        scores.append({
            "modelo":
                modelo,

            "score_relativo_medio":
                score,

            "lift_medio_percentual":
                (
                    score - 1
                )
                * 100,
        })

    return (
        resumo,
        pd.DataFrame(
            scores
        )
        .sort_values(
            "score_relativo_medio",
            ascending=False
        )
    )


# ============================================================
# RESUMO POR BLOCO
# ============================================================

def gerar_resumo_blocos(
    resultados
):

    return (
        resultados
        .groupby(
            [
                "bloco",
                "modelo",
                "qtd_exclusoes"
            ],
            as_index=False
        )
        .agg(
            media_acertos=(
                "acertos",
                "mean"
            )
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
        metricas,
        pesos
    ) = executar()

    (
        resumo,
        ranking_modelos
    ) = gerar_resumo(
        resultados
    )

    resumo_blocos = (
        gerar_resumo_blocos(
            resultados
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "RANKING DOS MODELOS"
    )

    print(
        "=" * 120
    )

    print(
        ranking_modelos
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "RESULTADOS POR EXCLUSÃO"
    )

    print(
        "=" * 120
    )

    print(
        resumo
        .sort_values(
            [
                "qtd_exclusoes",
                "media_acertos"
            ],
            ascending=[
                True,
                False
            ]
        )
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

        ranking_modelos.to_excel(
            writer,
            sheet_name="Ranking_Modelos",
            index=False
        )

        resumo.to_excel(
            writer,
            sheet_name="Resumo",
            index=False
        )

        resumo_blocos.to_excel(
            writer,
            sheet_name="Resumo_Blocos",
            index=False
        )

        metricas.to_excel(
            writer,
            sheet_name="Metricas",
            index=False
        )

        pesos.to_excel(
            writer,
            sheet_name="Pesos",
            index=False
        )

        resultados.to_excel(
            writer,
            sheet_name="Detalhes",
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