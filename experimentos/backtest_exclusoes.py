import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


# ============================================================
# IMPORTS DA RAIZ
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dados import carregar_resultados
from features import GeradorEstatisticasAvancadas


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ULTIMOS_CONCURSOS = 100

JANELA_MINIMA_TREINO = 200

# Durante desenvolvimento deixamos menor.
# Depois podemos voltar para 150 / 300.
N_ESTIMATORS = 50

MAX_DEPTH = 6

MIN_SAMPLES_LEAF = 5

SEED = 42


# Testaremos tudo na mesma execução.
CENARIOS_EXCLUSOES = [
    4,  # 21 candidatas
    5,  # 20 candidatas
    6,  # 19 candidatas
    7,  # 18 candidatas
]


ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "resultado_backtest_exclusoes.xlsx"
)


# ============================================================
# CRIA MODELO
# ============================================================

def criar_modelo():
    """
    Modelo MultiOutput atual.

    Para cada uma das 25 dezenas:

        1 = saiu
        0 = não saiu
    """

    random_forest = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced"
    )

    return MultiOutputClassifier(
        random_forest,
        n_jobs=-1
    )


# ============================================================
# RANKING DE NÃO SAIR
# ============================================================

def obter_ranking_nao_sair(
    modelo,
    features
):
    """
    Retorna as 25 dezenas ordenadas
    pela probabilidade estimada de NÃO sair.
    """

    features = np.asarray(
        features,
        dtype=np.float32
    ).reshape(1, -1)

    probabilidades = (
        modelo.predict_proba(
            features
        )
    )

    ranking = []

    for indice_dezena in range(25):

        estimador = (
            modelo.estimators_[
                indice_dezena
            ]
        )

        classes = (
            estimador.classes_
        )

        probs = (
            probabilidades[
                indice_dezena
            ][0]
        )

        # Queremos explicitamente P(classe=0)
        if 0 in classes:

            indice_classe_zero = (
                np.where(
                    classes == 0
                )[0][0]
            )

            prob_nao_sair = float(
                probs[
                    indice_classe_zero
                ]
            )

        else:

            prob_nao_sair = 0.0

        ranking.append({
            "dezena":
                indice_dezena + 1,

            "prob_nao_sair":
                prob_nao_sair
        })

    ranking.sort(
        key=lambda item:
            item["prob_nao_sair"],
        reverse=True
    )

    return ranking


# ============================================================
# UTILITÁRIOS
# ============================================================

def dezenas_para_texto(
    dezenas
):
    return " ".join(
        f"{dezena:02d}"
        for dezena
        in sorted(dezenas)
    )


def esperado_aleatorio(
    quantidade_exclusoes
):
    """
    Existem 10 dezenas não sorteadas
    dentro de um universo de 25.

    Portanto:

        E = N * 10/25
    """

    return (
        quantidade_exclusoes
        * (10 / 25)
    )


# ============================================================
# PREPARAÇÃO DO DATASET
# ============================================================

def preparar_dataset():
    """
    Carrega o Excel UMA VEZ.

    Calcula todas as features UMA VEZ.

    Retorna tudo em memória para o backtest.
    """

    print()
    print("Carregando histórico...")

    df, df_bolas = (
        carregar_resultados(
            ROOT
            / "lotofacil_resultados.xlsx"
        )
    )

    print(
        f"Concursos carregados: "
        f"{len(df_bolas)}"
    )

    print()
    print(
        "Inicializando gerador "
        "de estatísticas..."
    )

    gerador = (
        GeradorEstatisticasAvancadas(
            df_bolas
        )
    )

    print()
    print(
        "Pré-calculando TODAS as features..."
    )

    inicio = time.time()

    X_completo, y_completo, indices_completo = (
        gerador.construir_dataset_ml(
            janela_minima=
                JANELA_MINIMA_TREINO
        )
    )

    tempo = (
        time.time()
        - inicio
    )

    print(
        f"Features calculadas em "
        f"{tempo:.1f}s"
    )

    print(
        f"X completo: "
        f"{X_completo.shape}"
    )

    print(
        f"y completo: "
        f"{y_completo.shape}"
    )

    print(
        f"indices: "
        f"{indices_completo.shape}"
    )

    print()
    print(
        "Dataset carregado em memória."
    )

    return (
        df,
        gerador,
        X_completo,
        y_completo,
        indices_completo
    )


# ============================================================
# BACKTEST
# ============================================================

def executar_backtest():
    print("=" * 80)
    print(
        "BACKTEST DE EXCLUSÕES - LOTOFÁCIL"
    )
    print("=" * 80)

    (
        df,
        gerador,
        X_completo,
        y_completo,
        indices_completo
    ) = preparar_dataset()

    total_concursos = (
        gerador.total_sorteios
    )

    inicio_backtest = max(
        JANELA_MINIMA_TREINO + 1,
        total_concursos
        - ULTIMOS_CONCURSOS
    )

    quantidade_testes = (
        total_concursos
        - inicio_backtest
    )

    print()
    print(
        f"Concursos do backtest: "
        f"{quantidade_testes}"
    )

    print(
        f"Do índice "
        f"{inicio_backtest} "
        f"até "
        f"{total_concursos - 1}"
    )

    print()

    resultados = []

    inicio_total = time.time()

    # ========================================================
    # WALK-FORWARD
    # ========================================================

    for numero_teste, indice_alvo in enumerate(
        range(
            inicio_backtest,
            total_concursos
        ),
        start=1
    ):

        inicio_teste = time.time()

        # ====================================================
        # TREINO
        #
        # O target representado por indices_completo
        # deve ser estritamente anterior ao concurso alvo.
        # ====================================================

        mascara_treino = (
            indices_completo
            < indice_alvo
        )

        X_treino = (
            X_completo[
                mascara_treino
            ]
        )

        y_treino = (
            y_completo[
                mascara_treino
            ]
        )

        # ====================================================
        # FEATURES DO CONCURSO ALVO
        #
        # Já foram pré-calculadas!
        #
        # Nada é recalculado aqui.
        # ====================================================

        mascara_previsao = (
            indices_completo
            == indice_alvo
        )

        posicoes_previsao = (
            np.where(
                mascara_previsao
            )[0]
        )

        if len(posicoes_previsao) == 0:

            print(
                f"AVISO: não encontrei "
                f"features para índice "
                f"{indice_alvo}"
            )

            continue

        features_previsao = (
            X_completo[
                posicoes_previsao[0]
            ]
        )

        # ====================================================
        # TREINAMENTO
        # ====================================================

        modelo = (
            criar_modelo()
        )

        modelo.fit(
            X_treino,
            y_treino
        )

        # ====================================================
        # RANKING
        # ====================================================

        ranking = (
            obter_ranking_nao_sair(
                modelo,
                features_previsao
            )
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
            set(
                range(1, 26)
            )
            - sorteadas
        )

        # ====================================================
        # NÚMERO DO CONCURSO
        # ====================================================

        if "Concurso" in df.columns:

            concurso_real = int(
                df.iloc[
                    indice_alvo
                ][
                    "Concurso"
                ]
            )

        else:

            concurso_real = (
                indice_alvo + 1
            )

        # ====================================================
        # CENÁRIOS 4 / 5 / 6 / 7
        # ====================================================

        for qtd_exclusoes in (
            CENARIOS_EXCLUSOES
        ):

            exclusoes = set(
                item["dezena"]
                for item
                in ranking[
                    :qtd_exclusoes
                ]
            )

            acertos_exclusao = len(
                exclusoes
                & nao_sorteadas
            )

            erros_exclusao_dezenas = (
                exclusoes
                & sorteadas
            )

            candidatas = (
                set(
                    range(1, 26)
                )
                - exclusoes
            )

            sorteadas_preservadas = len(
                candidatas
                & sorteadas
            )

            resultados.append({
                "concurso":
                    concurso_real,

                "indice":
                    indice_alvo,

                "qtd_exclusoes":
                    qtd_exclusoes,

                "qtd_candidatas":
                    25 - qtd_exclusoes,

                "acertos_exclusao":
                    acertos_exclusao,

                "erros_exclusao":
                    len(
                        erros_exclusao_dezenas
                    ),

                "sorteadas_preservadas":
                    sorteadas_preservadas,

                "maximo_teorico":
                    sorteadas_preservadas,

                "exclusao_perfeita":
                    int(
                        acertos_exclusao
                        == qtd_exclusoes
                    ),

                "exclusoes":
                    dezenas_para_texto(
                        exclusoes
                    ),

                "erros_exclusao_dezenas":
                    dezenas_para_texto(
                        erros_exclusao_dezenas
                    ),

                "nao_sorteadas_reais":
                    dezenas_para_texto(
                        nao_sorteadas
                    ),

                "sorteadas_reais":
                    dezenas_para_texto(
                        sorteadas
                    ),
            })

        # ====================================================
        # PROGRESSO
        # ====================================================

        tempo_teste = (
            time.time()
            - inicio_teste
        )

        if (
            numero_teste == 1
            or numero_teste % 5 == 0
            or numero_teste
            == quantidade_testes
        ):

            tempo_total = (
                time.time()
                - inicio_total
            )

            media_por_teste = (
                tempo_total
                / numero_teste
            )

            print(
                f"Processados "
                f"{numero_teste}/"
                f"{quantidade_testes}"
                f" | último: "
                f"{tempo_teste:.2f}s"
                f" | média: "
                f"{media_por_teste:.2f}s"
            )

    return (
        pd.DataFrame(
            resultados
        )
    )


# ============================================================
# RESUMO
# ============================================================

def gerar_resumo(
    resultados
):
    linhas = []

    for qtd_exclusoes in (
        CENARIOS_EXCLUSOES
    ):

        dados = (
            resultados[
                resultados[
                    "qtd_exclusoes"
                ]
                == qtd_exclusoes
            ]
        )

        media_acertos = (
            dados[
                "acertos_exclusao"
            ]
            .mean()
        )

        esperado = (
            esperado_aleatorio(
                qtd_exclusoes
            )
        )

        ganho_absoluto = (
            media_acertos
            - esperado
        )

        if esperado > 0:

            ganho_percentual = (
                (
                    media_acertos
                    / esperado
                )
                - 1
            ) * 100

        else:

            ganho_percentual = 0

        exclusoes_perfeitas = int(
            dados[
                "exclusao_perfeita"
            ]
            .sum()
        )

        linhas.append({
            "qtd_exclusoes":
                qtd_exclusoes,

            "qtd_candidatas":
                25 - qtd_exclusoes,

            "concursos":
                len(dados),

            "media_acertos_exclusao":
                media_acertos,

            "aleatorio_esperado":
                esperado,

            "ganho_absoluto":
                ganho_absoluto,

            "ganho_percentual":
                ganho_percentual,

            "media_erros_exclusao":
                dados[
                    "erros_exclusao"
                ].mean(),

            "media_sorteadas_preservadas":
                dados[
                    "sorteadas_preservadas"
                ].mean(),

            "exclusoes_perfeitas":
                exclusoes_perfeitas,

            "percentual_perfeitas":
                (
                    exclusoes_perfeitas
                    / len(dados)
                    * 100
                    if len(dados) > 0
                    else 0
                )
        })

    return pd.DataFrame(
        linhas
    )


# ============================================================
# DISTRIBUIÇÃO
# ============================================================

def gerar_distribuicao(
    resultados
):
    linhas = []

    for qtd_exclusoes in (
        CENARIOS_EXCLUSOES
    ):

        dados = (
            resultados[
                resultados[
                    "qtd_exclusoes"
                ]
                == qtd_exclusoes
            ]
        )

        total = len(
            dados
        )

        for acertos in range(
            qtd_exclusoes + 1
        ):

            quantidade = int(
                (
                    dados[
                        "acertos_exclusao"
                    ]
                    == acertos
                )
                .sum()
            )

            linhas.append({
                "qtd_exclusoes":
                    qtd_exclusoes,

                "qtd_candidatas":
                    25 - qtd_exclusoes,

                "acertos_exclusao":
                    acertos,

                "quantidade_concursos":
                    quantidade,

                "percentual":
                    (
                        quantidade
                        / total
                        * 100
                        if total > 0
                        else 0
                    )
            })

    return pd.DataFrame(
        linhas
    )


# ============================================================
# MOSTRAR
# ============================================================

def mostrar_resultados(
    resumo,
    distribuicao
):
    print()
    print("=" * 100)
    print(
        "RESULTADO FINAL DO BACKTEST"
    )
    print("=" * 100)

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
    ]

    print(
        resumo[
            colunas
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    for qtd_exclusoes in (
        CENARIOS_EXCLUSOES
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"{qtd_exclusoes} EXCLUSÕES "
            f"→ "
            f"{25 - qtd_exclusoes} "
            f"CANDIDATAS"
        )

        print(
            "-" * 70
        )

        dados = (
            distribuicao[
                distribuicao[
                    "qtd_exclusoes"
                ]
                == qtd_exclusoes
            ]
        )

        print(
            dados[
                [
                    "acertos_exclusao",
                    "quantidade_concursos",
                    "percentual",
                ]
            ]
            .round(2)
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
    distribuicao
):
    ARQUIVO_SAIDA.parent.mkdir(
        parents=True,
        exist_ok=True
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
        "Arquivo gerado:"
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

    mostrar_resultados(
        resumo,
        distribuicao
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