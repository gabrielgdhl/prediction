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


from dados import carregar_resultados

from features_v2_reference import (
    GeradorFeaturesV2
)

from ranking_v3 import (
    selecionar_exclusoes_v3,
    obter_provaveis_estatisticos
)


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

ULTIMOS_CONCURSOS = 100

JANELA_MINIMA = 200

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


QTD_PROVAVEIS = 5


# ============================================================
# THRESHOLDS QUE VAMOS TESTAR
# ============================================================

MARGENS_PROTECAO = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15
]


LIFTS_EXAUSTAO = [
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08
]


AMOSTRAS_EXAUSTAO_MINIMAS = 30


ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "diagnostico_v3_thresholds.xlsx"
)


# ============================================================
# MODELO
# ============================================================

def criar_modelo():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )


# ============================================================
# HELPERS
# ============================================================

def esperado_aleatorio(qtd):
    return qtd * (10 / 25)


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
        resultado[int(dezena)] = {
            nome: float(valor)
            for nome, valor
            in zip(
                nomes,
                linha
            )
        }

    return resultado


def criar_ranking_exclusao(
    modelo,
    X_teste,
    dezenas_teste
):
    probabilidades = (
        modelo.predict_proba(
            X_teste
        )
    )

    indice_classe_1 = (
        np.where(
            modelo.classes_ == 1
        )[0][0]
    )

    prob_sair = (
        probabilidades[
            :,
            indice_classe_1
        ]
    )

    ranking = []

    for dezena, prob in zip(
        dezenas_teste,
        prob_sair
    ):
        prob = float(prob)

        ranking.append({
            "dezena": int(dezena),
            "prob_sair": prob,
            "prob_nao_sair": 1.0 - prob
        })

    ranking.sort(
        key=lambda item:
            item["prob_nao_sair"],
        reverse=True
    )

    return ranking


# ============================================================
# PREPARAÇÃO DO DATASET
# ============================================================

def preparar_dataset():
    print("=" * 100)
    print(
        "DIAGNÓSTICO V3 - "
        "MARGENS DE PROTEÇÃO E EXAUSTÃO"
    )
    print("=" * 100)

    print()
    print("Carregando histórico...")

    df, df_bolas = carregar_resultados(
        ROOT
        / "lotofacil_resultados.xlsx"
    )

    print(
        f"Concursos carregados: "
        f"{len(df_bolas)}"
    )

    gerador = GeradorFeaturesV2(
        df_bolas
    )

    print()
    print(
        "Construindo dataset V2..."
    )

    inicio = time.time()

    (
        X,
        y,
        indices_target,
        dezenas
    ) = gerador.construir_dataset(
        janela_minima=JANELA_MINIMA
    )

    print(
        f"Dataset criado em "
        f"{time.time() - inicio:.1f}s"
    )

    print(
        f"X = {X.shape}"
    )

    return (
        df,
        gerador,
        X,
        y,
        indices_target,
        dezenas
    )


# ============================================================
# CACHE DAS PREVISÕES DOS 100 CONCURSOS
#
# IMPORTANTE:
#
# Treinamos o Random Forest UMA VEZ por concurso.
#
# Depois testamos todos os thresholds em cima
# das mesmas previsões.
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
        gerador.total_sorteios
    )

    inicio_backtest = max(
        JANELA_MINIMA + 1,
        total - ULTIMOS_CONCURSOS
    )

    quantidade_testes = (
        total - inicio_backtest
    )

    print()
    print(
        f"Preparando {quantidade_testes} "
        f"casos walk-forward..."
    )

    casos = []

    inicio_total = time.time()

    for numero_teste, indice_alvo in enumerate(
        range(
            inicio_backtest,
            total
        ),
        start=1
    ):
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
            continue

        modelo = criar_modelo()

        modelo.fit(
            X_treino,
            y_treino
        )

        ranking_exclusao = (
            criar_ranking_exclusao(
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

        (
            provaveis,
            ranking_presenca
        ) = obter_provaveis_estatisticos(
            features_por_dezena=
                features_por_dezena,

            quantidade=
                QTD_PROVAVEIS
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

        casos.append({
            "concurso":
                concurso,

            "indice":
                indice_alvo,

            "ranking_exclusao":
                ranking_exclusao,

            "features_por_dezena":
                features_por_dezena,

            "provaveis":
                provaveis,

            "ranking_presenca":
                ranking_presenca,

            "sorteadas":
                sorteadas,

            "nao_sorteadas":
                nao_sorteadas
        })

        if (
            numero_teste == 1
            or numero_teste % 5 == 0
            or numero_teste
            == quantidade_testes
        ):
            print(
                f"{numero_teste:03d}/"
                f"{quantidade_testes}"
                f" | total="
                f"{time.time() - inicio_total:.1f}s"
            )

    return casos


# ============================================================
# EXECUTAR GRID DE THRESHOLDS
# ============================================================

def executar_grid(
    casos
):
    resultados = []

    total_combinacoes = (
        len(MARGENS_PROTECAO)
        * len(LIFTS_EXAUSTAO)
        * len(CENARIOS_EXCLUSOES)
    )

    contador = 0

    print()
    print(
        f"Testando "
        f"{total_combinacoes} combinações..."
    )

    # ========================================================
    # CADA COMBINAÇÃO DE PARÂMETROS
    # ========================================================

    for margem in MARGENS_PROTECAO:

        for lift_exaustao in (
            LIFTS_EXAUSTAO
        ):

            for qtd_exclusoes in (
                CENARIOS_EXCLUSOES
            ):

                contador += 1

                acertos_v2 = []

                acertos_v3 = []

                qtd_protegidas_total = 0

                protecoes_corretas = 0

                qtd_exaustoes_total = 0

                exaustoes_corretas = 0

                v3_melhor = 0
                v3_pior = 0
                v3_igual = 0

                perfeitas_v3 = 0

                # ============================================
                # RODAR OS 100 CASOS
                # ============================================

                for caso in casos:

                    ranking = (
                        caso[
                            "ranking_exclusao"
                        ]
                    )

                    provaveis = (
                        caso[
                            "provaveis"
                        ]
                    )

                    features = (
                        caso[
                            "features_por_dezena"
                        ]
                    )

                    sorteadas = (
                        caso[
                            "sorteadas"
                        ]
                    )

                    nao_sorteadas = (
                        caso[
                            "nao_sorteadas"
                        ]
                    )

                    # ----------------------------------------
                    # V2 BASELINE
                    # ----------------------------------------

                    exclusoes_v2 = {
                        item["dezena"]
                        for item
                        in ranking[
                            :qtd_exclusoes
                        ]
                    }

                    acerto_v2 = len(
                        exclusoes_v2
                        & nao_sorteadas
                    )

                    # ----------------------------------------
                    # V3
                    # ----------------------------------------

                    (
                        exclusoes_v3_lista,
                        detalhes
                    ) = (
                        selecionar_exclusoes_v3(
                            ranking_exclusao=
                                ranking,

                            provaveis=
                                provaveis,

                            features_por_dezena=
                                features,

                            quantidade_exclusoes=
                                qtd_exclusoes,

                            margem_minima=
                                margem,

                            lift_exaustao_minimo=
                                lift_exaustao,

                            amostras_exaustao_minimas=
                                AMOSTRAS_EXAUSTAO_MINIMAS
                        )
                    )

                    exclusoes_v3 = set(
                        exclusoes_v3_lista
                    )

                    acerto_v3 = len(
                        exclusoes_v3
                        & nao_sorteadas
                    )

                    acertos_v2.append(
                        acerto_v2
                    )

                    acertos_v3.append(
                        acerto_v3
                    )

                    # ----------------------------------------
                    # Comparação V3 vs V2
                    # ----------------------------------------

                    if acerto_v3 > acerto_v2:
                        v3_melhor += 1

                    elif acerto_v3 < acerto_v2:
                        v3_pior += 1

                    else:
                        v3_igual += 1

                    if (
                        acerto_v3
                        == qtd_exclusoes
                    ):
                        perfeitas_v3 += 1

                    # ========================================
                    # ANALISAR REGRAS INDIVIDUALMENTE
                    # ========================================

                    for detalhe in detalhes:

                        decisao = (
                            detalhe.get(
                                "decisao"
                            )
                        )

                        motivo = (
                            detalhe.get(
                                "motivo"
                            )
                        )

                        dezena = (
                            detalhe[
                                "dezena"
                            ]
                        )

                        # ------------------------------------
                        # PROTEÇÃO
                        #
                        # proteção correta significa:
                        # a pedra realmente saiu.
                        # ------------------------------------

                        if decisao == "PROTEGER":

                            qtd_protegidas_total += 1

                            if dezena in sorteadas:
                                protecoes_corretas += 1

                        # ------------------------------------
                        # EXAUSTÃO
                        #
                        # exaustão correta significa:
                        # pedra realmente NÃO saiu.
                        # ------------------------------------

                        if motivo == "EXAUSTAO_FORTE":

                            qtd_exaustoes_total += 1

                            if dezena in nao_sorteadas:
                                exaustoes_corretas += 1

                # ============================================
                # MÉTRICAS DA COMBINAÇÃO
                # ============================================

                media_v2 = float(
                    np.mean(
                        acertos_v2
                    )
                )

                media_v3 = float(
                    np.mean(
                        acertos_v3
                    )
                )

                aleatorio = (
                    esperado_aleatorio(
                        qtd_exclusoes
                    )
                )

                if qtd_protegidas_total > 0:

                    precisao_protecao = (
                        protecoes_corretas
                        / qtd_protegidas_total
                    )

                else:

                    precisao_protecao = np.nan

                if qtd_exaustoes_total > 0:

                    precisao_exaustao = (
                        exaustoes_corretas
                        / qtd_exaustoes_total
                    )

                else:

                    precisao_exaustao = np.nan

                resultados.append({
                    "margem_protecao":
                        margem,

                    "lift_exaustao":
                        lift_exaustao,

                    "qtd_exclusoes":
                        qtd_exclusoes,

                    "qtd_candidatas":
                        25 - qtd_exclusoes,

                    "concursos":
                        len(casos),

                    "media_v2":
                        media_v2,

                    "media_v3":
                        media_v3,

                    "aleatorio":
                        aleatorio,

                    "ganho_v3_vs_v2":
                        media_v3
                        - media_v2,

                    "ganho_v3_vs_aleatorio":
                        media_v3
                        - aleatorio,

                    "ganho_percentual_vs_aleatorio":
                        (
                            (
                                media_v3
                                / aleatorio
                            )
                            - 1
                        )
                        * 100,

                    "v3_melhor":
                        v3_melhor,

                    "v3_pior":
                        v3_pior,

                    "v3_igual":
                        v3_igual,

                    "protegidas_total":
                        qtd_protegidas_total,

                    "protecoes_corretas":
                        protecoes_corretas,

                    "precisao_protecao":
                        precisao_protecao,

                    "exaustoes_total":
                        qtd_exaustoes_total,

                    "exaustoes_corretas":
                        exaustoes_corretas,

                    "precisao_exaustao":
                        precisao_exaustao,

                    "perfeitas_v3":
                        perfeitas_v3
                })

                if (
                    contador == 1
                    or contador % 20 == 0
                    or contador
                    == total_combinacoes
                ):
                    print(
                        f"{contador}/"
                        f"{total_combinacoes}"
                    )

    return pd.DataFrame(
        resultados
    )


# ============================================================
# RANKING DAS MELHORES CONFIGURAÇÕES
# ============================================================

def gerar_ranking(
    resultados
):
    """
    Ordenação principal:

        maior media_v3
        depois maior ganho vs V2
        depois maior precisão das regras
    """

    ranking = (
        resultados
        .copy()
    )

    ranking[
        "score_regras"
    ] = (
        ranking[
            "precisao_protecao"
        ]
        .fillna(0)
        +
        ranking[
            "precisao_exaustao"
        ]
        .fillna(0)
    )

    ranking = (
        ranking
        .sort_values(
            [
                "qtd_exclusoes",
                "media_v3",
                "ganho_v3_vs_v2",
                "score_regras"
            ],
            ascending=[
                True,
                False,
                False,
                False
            ]
        )
    )

    return ranking


# ============================================================
# MELHOR POR CENÁRIO
# ============================================================

def melhores_por_cenario(
    ranking
):
    linhas = []

    for qtd in CENARIOS_EXCLUSOES:

        dados = (
            ranking[
                ranking[
                    "qtd_exclusoes"
                ]
                == qtd
            ]
        )

        if len(dados) == 0:
            continue

        linhas.append(
            dados.iloc[0]
        )

    return pd.DataFrame(
        linhas
    )


# ============================================================
# OUTPUT
# ============================================================

def mostrar(
    melhores
):
    print()
    print("=" * 130)
    print(
        "MELHORES THRESHOLDS V3 "
        "POR QUANTIDADE DE EXCLUSÕES"
    )
    print("=" * 130)

    colunas = [
        "qtd_exclusoes",
        "qtd_candidatas",

        "margem_protecao",
        "lift_exaustao",

        "media_v2",
        "media_v3",
        "aleatorio",

        "ganho_v3_vs_v2",
        "ganho_percentual_vs_aleatorio",

        "v3_melhor",
        "v3_pior",
        "v3_igual",

        "protegidas_total",
        "precisao_protecao",

        "exaustoes_total",
        "precisao_exaustao",

        "perfeitas_v3"
    ]

    print(
        melhores[
            colunas
        ]
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
    ranking,
    melhores
):
    with pd.ExcelWriter(
        ARQUIVO_SAIDA,
        engine="openpyxl"
    ) as writer:

        melhores.to_excel(
            writer,
            sheet_name="Melhores",
            index=False
        )

        ranking.to_excel(
            writer,
            sheet_name="Ranking",
            index=False
        )

        resultados.to_excel(
            writer,
            sheet_name="Todos",
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
    inicio = time.time()

    casos = (
        preparar_casos()
    )

    resultados = (
        executar_grid(
            casos
        )
    )

    ranking = (
        gerar_ranking(
            resultados
        )
    )

    melhores = (
        melhores_por_cenario(
            ranking
        )
    )

    mostrar(
        melhores
    )

    exportar(
        resultados,
        ranking,
        melhores
    )

    print()
    print(
        f"Tempo total: "
        f"{time.time() - inicio:.1f}s"
    )


if __name__ == "__main__":
    main()