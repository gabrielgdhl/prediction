import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================
# IMPORTS DO PROJETO
# ============================================================

from dados import carregar_resultados

from cache_dataset import obter_dataset_v2

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
    CENARIOS_EXCLUSOES,
    GRUPO_RANKING_V2,
    GRUPO_FREQUENCIA_MICRO,
    GRUPO_FREQUENCIA_MESO,
    GRUPO_BASELINE_LONGO,
    GRUPO_TENDENCIA_MICRO,
    GRUPO_MUDANCA_REGIME,
    GRUPO_TENDENCIA_LONGA,
)


# ============================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# ============================================================

TAMANHOS_META_TREINO = list(
    range(3, 101)
)

BLOCOS_TESTE = 4

TAMANHO_BLOCO_TESTE = 100


# ============================================================
# MULTIESCALA
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
# SAÍDA
# ============================================================

ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "resultado_memoria_multiescala.xlsx"
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

def esperado_aleatorio(qtd):
    return qtd * (10 / 25)


def criar_features_por_dezena(
    X_teste,
    dezenas_teste,
):
    nomes = GeradorFeaturesV2.nomes_features()

    resultado = {}

    for linha, dezena in zip(
        X_teste,
        dezenas_teste,
    ):
        resultado[int(dezena)] = {
            nome: float(valor)
            for nome, valor in zip(
                nomes,
                linha,
            )
        }

    return resultado


# ============================================================
# RANKING V2
# ============================================================

def criar_ranking_v2(
    modelo,
    X_teste,
    dezenas_teste,
):
    probabilidades = modelo.predict_proba(
        X_teste
    )

    indice_classe_1 = int(
        np.where(
            modelo.classes_ == 1
        )[0][0]
    )

    probs_sair = probabilidades[
        :,
        indice_classe_1
    ]

    ranking = []

    for dezena, prob_sair in zip(
        dezenas_teste,
        probs_sair,
    ):
        prob_sair = float(prob_sair)

        ranking.append({
            "dezena": int(dezena),
            "prob_sair": prob_sair,
            "prob_nao_sair": 1.0 - prob_sair,
        })

    ranking.sort(
        key=lambda item:
            item["prob_nao_sair"],
        reverse=True,
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
    print("TESTE DE MEMÓRIA DO MODELO MULTIESCALA")
    print("=" * 100)

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
        matriz_binaria,
    ) = obter_dataset_v2(
        caminho_excel=caminho_excel,
        caminho_features=caminho_features,
        df_bolas=df_bolas,
        classe_gerador=GeradorFeaturesV2,
        janela_minima=JANELA_MINIMA,
    )

    if gerador is None:

        class GeradorCache:
            pass

        gerador = GeradorCache()

        gerador.total_sorteios = len(
            matriz_binaria
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
# GERAR CASO WALK-FORWARD
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

    X_treino = X[
        mascara_treino
    ]

    y_treino = y[
        mascara_treino
    ]

    X_teste = X[
        mascara_teste
    ]

    dezenas_teste = dezenas[
        mascara_teste
    ]

    if len(X_teste) != 25:
        raise ValueError(
            f"Índice {indice_alvo}: "
            f"{len(X_teste)} dezenas."
        )

    modelo_v2 = criar_modelo_v2()

    modelo_v2.fit(
        X_treino,
        y_treino,
    )

    ranking_v2 = criar_ranking_v2(
        modelo_v2,
        X_teste,
        dezenas_teste,
    )

    features_por_dezena = (
        criar_features_por_dezena(
            X_teste,
            dezenas_teste,
        )
    )

    extras_v5 = (
        calcular_features_v5_concurso(
            matriz_binaria=
                gerador.matriz_binaria,

            indice_estado=
                indice_alvo - 1,
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
        set(range(1, 26))
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
# PREPARAR TODOS OS CASOS
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

    total = gerador.total_sorteios

    maior_treino = max(
        TAMANHOS_META_TREINO
    )

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
        - maior_treino
    )

    primeiro_indice = max(
        primeiro_indice,
        JANELA_MINIMA + 1,
    )

    indices = list(
        range(
            primeiro_indice,
            total,
        )
    )

    print()
    print(
        f"Gerando casos de "
        f"{primeiro_indice} "
        f"até {total - 1}"
    )

    print(
        f"Total de casos: "
        f"{len(indices)}"
    )

    casos = {}

    inicio = time.time()

    for numero, indice in enumerate(
        indices,
        start=1,
    ):
        casos[indice] = gerar_caso(
            indice_alvo=indice,
            df=df,
            gerador=gerador,
            X=X,
            y=y,
            indices_target=indices_target,
            dezenas=dezenas,
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
        total,
    )


# ============================================================
# DATASET META
# ============================================================

def construir_dataset_meta(
    casos,
    indices_meta,
):
    X_meta = []
    y_meta = []

    for indice in indices_meta:

        caso = casos[indice]

        matriz = construir_matriz_custom(
            ranking_v2=
                caso["ranking_v2"],

            features_por_dezena=
                caso[
                    "features_por_dezena"
                ],

            extras_por_dezena=
                caso["extras_v5"],

            features_extras=
                FEATURES_MULTIESCALA,
        )

        for posicao in range(25):

            dezena = (
                posicao + 1
            )

            X_meta.append(
                matriz[posicao]
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
            dtype=np.float64,
        ),

        np.asarray(
            y_meta,
            dtype=np.int8,
        ),
    )


# ============================================================
# TESTAR
# ============================================================

def testar_modelo(
    modelo,
    casos,
    indices_teste,
    bloco,
    tamanho_treino,
):
    resultados = []

    for indice in indices_teste:

        caso = casos[indice]

        ranking = criar_ranking_custom(
            modelo=modelo,

            ranking_v2=
                caso["ranking_v2"],

            features_por_dezena=
                caso[
                    "features_por_dezena"
                ],

            extras_por_dezena=
                caso["extras_v5"],

            features_extras=
                FEATURES_MULTIESCALA,
        )

        for qtd in CENARIOS_EXCLUSOES:

            exclusoes = {
                item["dezena"]
                for item
                in ranking[:qtd]
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

                "tamanho_treino":
                    tamanho_treino,

                "concurso":
                    caso[
                        "concurso"
                    ],

                "qtd_exclusoes":
                    qtd,

                "acertos":
                    acertos,
            })

    return resultados


# ============================================================
# EXECUTAR
# ============================================================

def executar():
    (
        casos,
        total,
    ) = preparar_casos()

    resultados = []

    # ========================================================
    # BLOCOS
    # ========================================================

    for bloco in range(
        1,
        BLOCOS_TESTE + 1,
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

        indices_teste = list(
            range(
                inicio_teste,
                fim_teste,
            )
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

        # ====================================================
        # TAMANHOS DE MEMÓRIA
        # ====================================================

        for tamanho_treino in (
            TAMANHOS_META_TREINO
        ):

            fim_meta = (
                inicio_teste
            )

            inicio_meta = (
                fim_meta
                - tamanho_treino
            )

            indices_meta = list(
                range(
                    inicio_meta,
                    fim_meta,
                )
            )

            print(
                f"  → treino "
                f"{tamanho_treino}"
                f" | "
                f"{inicio_meta}.."
                f"{fim_meta - 1}"
            )

            (
                X_meta,
                y_meta,
            ) = construir_dataset_meta(
                casos=
                    casos,

                indices_meta=
                    indices_meta,
            )

            modelo = treinar_modelo_meta(
                X_meta,
                y_meta,
            )

            resultados.extend(
                testar_modelo(
                    modelo=
                        modelo,

                    casos=
                        casos,

                    indices_teste=
                        indices_teste,

                    bloco=
                        bloco,

                    tamanho_treino=
                        tamanho_treino,
                )
            )

    return pd.DataFrame(
        resultados
    )


# ============================================================
# RESUMO POR TAMANHO
# ============================================================

def gerar_resumo(
    resultados,
):
    resumo = (
        resultados
        .groupby(
            [
                "tamanho_treino",
                "qtd_exclusoes",
            ],
            as_index=False,
        )
        .agg(
            concursos=(
                "concurso",
                "count",
            ),

            media_acertos=(
                "acertos",
                "mean",
            ),

            desvio_acertos=(
                "acertos",
                "std",
            ),
        )
    )

    resumo[
        "aleatorio"
    ] = (
        resumo[
            "qtd_exclusoes"
        ]
        .apply(
            esperado_aleatorio
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
        "lift_percentual"
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

    return resumo


# ============================================================
# SCORE AGREGADO
# ============================================================

def gerar_ranking_memoria(
    resumo,
):
    dados = resumo.copy()

    dados[
        "score_relativo"
    ] = (
        dados[
            "media_acertos"
        ]
        / dados[
            "aleatorio"
        ]
    )

    ranking = (
        dados
        .groupby(
            "tamanho_treino",
            as_index=False,
        )
        .agg(
            score_relativo_medio=(
                "score_relativo",
                "mean",
            )
        )
    )

    ranking[
        "lift_medio_percentual"
    ] = (
        (
            ranking[
                "score_relativo_medio"
            ]
            - 1
        )
        * 100
    )

    ranking = (
        ranking
        .sort_values(
            "score_relativo_medio",
            ascending=False,
        )
    )

    return ranking


# ============================================================
# SCORE POR BLOCO
# ============================================================

def gerar_resumo_blocos(
    resultados,
):
    resumo = (
        resultados
        .groupby(
            [
                "bloco",
                "tamanho_treino",
                "qtd_exclusoes",
            ],
            as_index=False,
        )
        .agg(
            media_acertos=(
                "acertos",
                "mean",
            )
        )
    )

    resumo[
        "aleatorio"
    ] = (
        resumo[
            "qtd_exclusoes"
        ]
        .apply(
            esperado_aleatorio
        )
    )

    resumo[
        "score_relativo"
    ] = (
        resumo[
            "media_acertos"
        ]
        / resumo[
            "aleatorio"
        ]
    )

    score_blocos = (
        resumo
        .groupby(
            [
                "bloco",
                "tamanho_treino",
            ],
            as_index=False,
        )
        .agg(
            score_relativo_medio=(
                "score_relativo",
                "mean",
            )
        )
    )

    score_blocos[
        "lift_medio_percentual"
    ] = (
        (
            score_blocos[
                "score_relativo_medio"
            ]
            - 1
        )
        * 100
    )

    return (
        resumo,
        score_blocos,
    )


# ============================================================
# MATRIZ DE ROBUSTEZ
# ============================================================

def gerar_matriz_robustez(
    score_blocos,
):
    matriz = (
        score_blocos
        .pivot(
            index=
                "tamanho_treino",

            columns=
                "bloco",

            values=
                "lift_medio_percentual",
        )
        .reset_index()
    )

    colunas_blocos = [
        coluna
        for coluna in matriz.columns
        if coluna != "tamanho_treino"
    ]

    matriz[
        "media_lift"
    ] = (
        matriz[
            colunas_blocos
        ]
        .mean(axis=1)
    )

    matriz[
        "min_lift"
    ] = (
        matriz[
            colunas_blocos
        ]
        .min(axis=1)
    )

    matriz[
        "max_lift"
    ] = (
        matriz[
            colunas_blocos
        ]
        .max(axis=1)
    )

    matriz[
        "desvio_lift"
    ] = (
        matriz[
            colunas_blocos
        ]
        .std(axis=1)
    )

    matriz[
        "blocos_positivos"
    ] = (
        (
            matriz[
                colunas_blocos
            ]
            > 0
        )
        .sum(axis=1)
    )

    matriz[
        "score_robustez"
    ] = (
        matriz[
            "media_lift"
        ]
        + (
            0.25
            * matriz[
                "min_lift"
            ]
        )
        - (
            0.25
            * matriz[
                "desvio_lift"
            ]
        )
    )

    return (
        matriz
        .sort_values(
            "score_robustez",
            ascending=False,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    inicio = time.time()

    resultados = executar()

    resumo = gerar_resumo(
        resultados
    )

    ranking = (
        gerar_ranking_memoria(
            resumo
        )
    )

    (
        resumo_blocos,
        score_blocos,
    ) = gerar_resumo_blocos(
        resultados
    )

    robustez = (
        gerar_matriz_robustez(
            score_blocos
        )
    )

    print()
    print("=" * 120)
    print("RANKING DOS TAMANHOS DE MEMÓRIA")
    print("=" * 120)

    print(
        ranking
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("ROBUSTEZ TEMPORAL")
    print("=" * 120)

    print(
        robustez
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("RESULTADOS POR EXCLUSÃO")
    print("=" * 120)

    print(
        resumo
        .sort_values(
            [
                "qtd_exclusoes",
                "media_acertos",
            ],
            ascending=[
                True,
                False,
            ],
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
        engine="openpyxl",
    ) as writer:

        ranking.to_excel(
            writer,
            sheet_name="Ranking_Memoria",
            index=False,
        )

        robustez.to_excel(
            writer,
            sheet_name="Robustez",
            index=False,
        )

        resumo.to_excel(
            writer,
            sheet_name="Resumo",
            index=False,
        )

        score_blocos.to_excel(
            writer,
            sheet_name="Score_Blocos",
            index=False,
        )

        resumo_blocos.to_excel(
            writer,
            sheet_name="Bloco_Exclusao",
            index=False,
        )

        resultados.to_excel(
            writer,
            sheet_name="Detalhes",
            index=False,
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