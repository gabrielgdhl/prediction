import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from dados import carregar_resultados
from features_v2 import GeradorFeaturesV2


# ============================================================
# CONFIGURAÇÕES
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


ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "resultado_backtest_exclusoes_v2.xlsx"
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
# UTILITÁRIOS
# ============================================================

def dezenas_para_texto(dezenas):

    return " ".join(
        f"{d:02d}"
        for d in sorted(dezenas)
    )


def esperado_aleatorio(qtd):

    return qtd * (10 / 25)


# ============================================================
# DATASET
# ============================================================

def preparar_dataset():

    print("=" * 80)
    print("BACKTEST DE EXCLUSÕES V2")
    print("=" * 80)

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

    print()
    print(
        f"X: {X.shape}"
    )

    print(
        f"y: {y.shape}"
    )

    print(
        f"Observações: {len(y):,}"
    )

    print(
        f"Features por observação: "
        f"{X.shape[1]}"
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
# BACKTEST
# ============================================================

def executar_backtest():

    (
        df,
        gerador,
        X,
        y,
        indices_target,
        dezenas
    ) = preparar_dataset()

    total = gerador.total_sorteios

    inicio_backtest = max(
        JANELA_MINIMA + 1,
        total - ULTIMOS_CONCURSOS
    )

    resultados = []

    print()
    print(
        f"Executando walk-forward "
        f"nos últimos "
        f"{total - inicio_backtest} "
        f"concursos..."
    )

    inicio_total = time.time()

    for numero_teste, indice_alvo in enumerate(
        range(
            inicio_backtest,
            total
        ),
        start=1
    ):

        inicio_teste = time.time()

        # ====================================================
        # TREINO
        #
        # Somente concursos anteriores.
        # ====================================================

        mascara_treino = (
            indices_target
            < indice_alvo
        )

        X_treino = X[
            mascara_treino
        ]

        y_treino = y[
            mascara_treino
        ]

        # ====================================================
        # TESTE
        #
        # Exatamente 25 linhas:
        # uma para cada dezena.
        # ====================================================

        mascara_teste = (
            indices_target
            == indice_alvo
        )

        X_teste = X[
            mascara_teste
        ]

        dezenas_teste = dezenas[
            mascara_teste
        ]

        if len(X_teste) != 25:

            print(
                f"AVISO: índice "
                f"{indice_alvo} possui "
                f"{len(X_teste)} linhas."
            )

            continue

        # ====================================================
        # TREINA UM ÚNICO MODELO
        # ====================================================

        modelo = criar_modelo()

        modelo.fit(
            X_treino,
            y_treino
        )

        # ====================================================
        # P(SAIR)
        # ====================================================

        probabilidades = (
            modelo.predict_proba(
                X_teste
            )
        )

        classes = modelo.classes_

        indice_classe_um = (
            np.where(
                classes == 1
            )[0][0]
        )

        prob_sair = (
            probabilidades[
                :,
                indice_classe_um
            ]
        )

        # Queremos excluir quem tem
        # MENOR probabilidade de sair.
        ranking = []

        for dezena, prob in zip(
            dezenas_teste,
            prob_sair
        ):

            ranking.append({
                "dezena":
                    int(dezena),

                "prob_sair":
                    float(prob),

                "prob_nao_sair":
                    float(
                        1 - prob
                    )
            })

        ranking.sort(
            key=lambda item:
                item["prob_sair"]
        )

        # ====================================================
        # RESULTADO REAL
        # ====================================================

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

        # ====================================================
        # TESTA 4/5/6/7 EXCLUSÕES
        # ====================================================

        for qtd in CENARIOS_EXCLUSOES:

            top = ranking[:qtd]

            exclusoes = {
                item["dezena"]
                for item in top
            }

            acertos = len(
                exclusoes
                & nao_sorteadas
            )

            erros_dezenas = (
                exclusoes
                & sorteadas
            )

            preservadas = (
                15
                - len(erros_dezenas)
            )

            media_prob_nao_sair = float(
                np.mean([
                    item["prob_nao_sair"]
                    for item in top
                ])
            )

            menor_prob_nao_sair = float(
                min(
                    item["prob_nao_sair"]
                    for item in top
                )
            )

            maior_prob_nao_sair = float(
                max(
                    item["prob_nao_sair"]
                    for item in top
                )
            )

            resultados.append({
                "concurso":
                    concurso,

                "indice":
                    indice_alvo,

                "qtd_exclusoes":
                    qtd,

                "qtd_candidatas":
                    25 - qtd,

                "acertos_exclusao":
                    acertos,

                "erros_exclusao":
                    len(erros_dezenas),

                "sorteadas_preservadas":
                    preservadas,

                "exclusao_perfeita":
                    int(
                        acertos == qtd
                    ),

                "media_prob_nao_sair":
                    media_prob_nao_sair,

                "menor_prob_nao_sair":
                    menor_prob_nao_sair,

                "maior_prob_nao_sair":
                    maior_prob_nao_sair,

                "exclusoes":
                    dezenas_para_texto(
                        exclusoes
                    ),

                "erros_exclusao_dezenas":
                    dezenas_para_texto(
                        erros_dezenas
                    ),

                "sorteadas_reais":
                    dezenas_para_texto(
                        sorteadas
                    ),

                "nao_sorteadas_reais":
                    dezenas_para_texto(
                        nao_sorteadas
                    )
            })

        if (
            numero_teste == 1
            or numero_teste % 5 == 0
            or indice_alvo == total - 1
        ):

            tempo = (
                time.time()
                - inicio_teste
            )

            total_decorrido = (
                time.time()
                - inicio_total
            )

            print(
                f"{numero_teste:03d}/"
                f"{total - inicio_backtest}"
                f" | último={tempo:.2f}s"
                f" | total="
                f"{total_decorrido:.1f}s"
            )

    return pd.DataFrame(
        resultados
    )


# ============================================================
# RESUMO
# ============================================================

def gerar_resumo(resultados):

    linhas = []

    for qtd in CENARIOS_EXCLUSOES:

        dados = resultados[
            resultados[
                "qtd_exclusoes"
            ] == qtd
        ]

        media = (
            dados[
                "acertos_exclusao"
            ].mean()
        )

        esperado = (
            esperado_aleatorio(qtd)
        )

        perfeitas = int(
            dados[
                "exclusao_perfeita"
            ].sum()
        )

        linhas.append({
            "qtd_exclusoes":
                qtd,

            "qtd_candidatas":
                25 - qtd,

            "concursos":
                len(dados),

            "media_acertos_exclusao":
                media,

            "aleatorio_esperado":
                esperado,

            "ganho_absoluto":
                media - esperado,

            "ganho_percentual":
                (
                    (
                        media
                        / esperado
                    ) - 1
                ) * 100,

            "media_erros_exclusao":
                dados[
                    "erros_exclusao"
                ].mean(),

            "media_sorteadas_preservadas":
                dados[
                    "sorteadas_preservadas"
                ].mean(),

            "exclusoes_perfeitas":
                perfeitas,

            "percentual_perfeitas":
                (
                    perfeitas
                    / len(dados)
                    * 100
                ),

            "media_confianca_exclusoes":
                dados[
                    "media_prob_nao_sair"
                ].mean()
        })

    return pd.DataFrame(
        linhas
    )


# ============================================================
# DISTRIBUIÇÃO
# ============================================================

def gerar_distribuicao(resultados):

    linhas = []

    for qtd in CENARIOS_EXCLUSOES:

        dados = resultados[
            resultados[
                "qtd_exclusoes"
            ] == qtd
        ]

        total = len(dados)

        for acertos in range(
            qtd + 1
        ):

            quantidade = int(
                (
                    dados[
                        "acertos_exclusao"
                    ]
                    == acertos
                ).sum()
            )

            linhas.append({
                "qtd_exclusoes":
                    qtd,

                "acertos":
                    acertos,

                "concursos":
                    quantidade,

                "percentual":
                    (
                        quantidade
                        / total
                        * 100
                    )
            })

    return pd.DataFrame(
        linhas
    )


# ============================================================
# OUTPUT
# ============================================================

def mostrar_resumo(resumo):

    print()
    print("=" * 110)
    print("RESULTADO V2")
    print("=" * 110)

    colunas = [
        "qtd_exclusoes",
        "qtd_candidatas",
        "media_acertos_exclusao",
        "aleatorio_esperado",
        "ganho_absoluto",
        "ganho_percentual",
        "media_sorteadas_preservadas",
        "exclusoes_perfeitas",
        "percentual_perfeitas",
        "media_confianca_exclusoes"
    ]

    print(
        resumo[
            colunas
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
    distribuicao
):

    with pd.ExcelWriter(
        ARQUIVO_SAIDA,
        engine="openpyxl"
    ) as writer:

        resumo.to_excel(
            writer,
            sheet_name="Resumo",
            index=False
        )

        distribuicao.to_excel(
            writer,
            sheet_name="Distribuicao",
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

    inicio = time.time()

    resultados = (
        executar_backtest()
    )

    resumo = (
        gerar_resumo(
            resultados
        )
    )

    distribuicao = (
        gerar_distribuicao(
            resultados
        )
    )

    mostrar_resumo(
        resumo
    )

    exportar(
        resultados,
        resumo,
        distribuicao
    )

    print()
    print(
        f"Tempo total: "
        f"{time.time() - inicio:.1f}s"
    )


if __name__ == "__main__":
    main()