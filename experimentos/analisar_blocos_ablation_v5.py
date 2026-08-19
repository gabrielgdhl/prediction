from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ARQUIVO_ENTRADA = (
    ROOT
    / "experimentos"
    / "resultado_ablation_v5.xlsx"
)

ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "analise_blocos_ablation_v5.xlsx"
)


# ============================================================
# CONFIG
# ============================================================

MODELOS_PRINCIPAIS = [
    "V4",
    "V4_RANK_TEND_FREQ_ANTERIOR",
    "V4_MICRO_LONGO",
    "V4_MULTIESCALA",
]


# ============================================================
# HELPERS
# ============================================================

def esperado_aleatorio(
    qtd_exclusoes
):
    return (
        qtd_exclusoes
        * (10 / 25)
    )


# ============================================================
# CARREGAR
# ============================================================

def carregar_dados():

    if not ARQUIVO_ENTRADA.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado: "
            f"{ARQUIVO_ENTRADA}"
        )

    print(
        "=" * 100
    )

    print(
        "ANÁLISE TEMPORAL DOS BLOCOS - ABLATION V5"
    )

    print(
        "=" * 100
    )

    print()
    print(
        f"Lendo:"
    )

    print(
        ARQUIVO_ENTRADA
    )

    detalhes = pd.read_excel(
        ARQUIVO_ENTRADA,
        sheet_name="Detalhes"
    )

    colunas_necessarias = {
        "bloco",
        "modelo",
        "concurso",
        "qtd_exclusoes",
        "acertos",
    }

    faltantes = (
        colunas_necessarias
        - set(
            detalhes.columns
        )
    )

    if faltantes:

        raise ValueError(
            "Colunas ausentes em Detalhes: "
            + ", ".join(
                sorted(
                    faltantes
                )
            )
        )

    print()
    print(
        f"Registros carregados: "
        f"{len(detalhes)}"
    )

    print(
        f"Blocos: "
        f"{sorted(detalhes['bloco'].unique())}"
    )

    return detalhes


# ============================================================
# RESUMO POR BLOCO + EXCLUSÃO
# ============================================================

def gerar_resumo_blocos(
    detalhes
):

    resumo = (
        detalhes
        .groupby(
            [
                "bloco",
                "modelo",
                "qtd_exclusoes",
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
# SCORE POR BLOCO
#
# Agrega Top4 / Top5 / Top6 / Top7.
#
# Fazemos:
#
# média_acertos / aleatório
#
# e depois média entre os quatro cortes.
# ============================================================

def gerar_score_por_bloco(
    resumo_blocos
):

    dados = (
        resumo_blocos
        .copy()
    )

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

    score = (
        dados
        .groupby(
            [
                "bloco",
                "modelo",
            ],
            as_index=False
        )
        .agg(
            score_relativo_medio=(
                "score_relativo",
                "mean"
            ),

            media_ganho_absoluto=(
                "ganho_absoluto",
                "mean"
            ),
        )
    )

    score[
        "lift_medio_percentual"
    ] = (
        (
            score[
                "score_relativo_medio"
            ]
            - 1
        )
        * 100
    )

    return score


# ============================================================
# MATRIZ:
#
#                 BLOCO 1   BLOCO 2   BLOCO 3   BLOCO 4
#
# V4
# MICRO_LONGO
# MULTIESCALA
# ...
# ============================================================

def gerar_matriz_lift(
    score_blocos
):

    matriz = (
        score_blocos
        .pivot(
            index="modelo",
            columns="bloco",
            values="lift_medio_percentual"
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Ordenação das colunas
    # --------------------------------------------------------

    colunas_bloco = sorted(
        [
            coluna
            for coluna
            in matriz.columns
            if coluna != "modelo"
        ]
    )

    matriz = matriz[
        [
            "modelo",
            *colunas_bloco,
        ]
    ]

    # --------------------------------------------------------
    # Estatísticas de estabilidade
    # --------------------------------------------------------

    matriz[
        "media_lift"
    ] = (
        matriz[
            colunas_bloco
        ]
        .mean(
            axis=1
        )
    )

    matriz[
        "min_lift"
    ] = (
        matriz[
            colunas_bloco
        ]
        .min(
            axis=1
        )
    )

    matriz[
        "max_lift"
    ] = (
        matriz[
            colunas_bloco
        ]
        .max(
            axis=1
        )
    )

    matriz[
        "desvio_lift"
    ] = (
        matriz[
            colunas_bloco
        ]
        .std(
            axis=1
        )
    )

    matriz[
        "blocos_positivos"
    ] = (
        (
            matriz[
                colunas_bloco
            ]
            > 0
        )
        .sum(
            axis=1
        )
    )

    matriz[
        "blocos_negativos"
    ] = (
        (
            matriz[
                colunas_bloco
            ]
            < 0
        )
        .sum(
            axis=1
        )
    )

    # ========================================================
    # SCORE DE ROBUSTEZ
    #
    # Queremos:
    #
    # média alta
    # +
    # mínimo não muito ruim
    # -
    # instabilidade
    #
    # Não é usado para treinar nada.
    # Serve apenas para diagnóstico.
    # ========================================================

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

    matriz = (
        matriz
        .sort_values(
            "score_robustez",
            ascending=False
        )
    )

    return matriz


# ============================================================
# RESULTADO POR CORTE
#
# Mostra estabilidade específica para:
#
# Top4
# Top5
# Top6
# Top7
# ============================================================

def gerar_matriz_por_exclusao(
    resumo_blocos,
    qtd_exclusoes
):

    dados = (
        resumo_blocos[
            resumo_blocos[
                "qtd_exclusoes"
            ]
            == qtd_exclusoes
        ]
        .copy()
    )

    matriz = (
        dados
        .pivot(
            index="modelo",
            columns="bloco",
            values="lift_percentual"
        )
        .reset_index()
    )

    colunas_bloco = sorted(
        [
            coluna
            for coluna
            in matriz.columns
            if coluna != "modelo"
        ]
    )

    matriz[
        "media"
    ] = (
        matriz[
            colunas_bloco
        ]
        .mean(
            axis=1
        )
    )

    matriz[
        "minimo"
    ] = (
        matriz[
            colunas_bloco
        ]
        .min(
            axis=1
        )
    )

    matriz[
        "maximo"
    ] = (
        matriz[
            colunas_bloco
        ]
        .max(
            axis=1
        )
    )

    matriz[
        "desvio"
    ] = (
        matriz[
            colunas_bloco
        ]
        .std(
            axis=1
        )
    )

    matriz[
        "positivos"
    ] = (
        (
            matriz[
                colunas_bloco
            ]
            > 0
        )
        .sum(
            axis=1
        )
    )

    matriz = (
        matriz
        .sort_values(
            "media",
            ascending=False
        )
    )

    return matriz


# ============================================================
# APENAS MODELOS PRINCIPAIS
# ============================================================

def filtrar_principais(
    dataframe
):

    if (
        "modelo"
        not in dataframe.columns
    ):
        return dataframe

    return (
        dataframe[
            dataframe[
                "modelo"
            ]
            .isin(
                MODELOS_PRINCIPAIS
            )
        ]
        .copy()
    )


# ============================================================
# COMPARAR DIRETAMENTE COM V4
# ============================================================

def gerar_comparacao_vs_v4(
    score_blocos
):

    v4 = (
        score_blocos[
            score_blocos[
                "modelo"
            ]
            == "V4"
        ][
            [
                "bloco",
                "lift_medio_percentual",
            ]
        ]
        .rename(
            columns={
                "lift_medio_percentual":
                    "lift_v4"
            }
        )
    )

    outros = (
        score_blocos[
            score_blocos[
                "modelo"
            ]
            != "V4"
        ]
        .copy()
    )

    comparacao = (
        outros
        .merge(
            v4,
            on="bloco",
            how="left"
        )
    )

    comparacao[
        "ganho_vs_v4"
    ] = (
        comparacao[
            "lift_medio_percentual"
        ]
        - comparacao[
            "lift_v4"
        ]
    )

    resumo = (
        comparacao
        .groupby(
            "modelo",
            as_index=False
        )
        .agg(
            ganho_medio_vs_v4=(
                "ganho_vs_v4",
                "mean"
            ),

            menor_ganho_vs_v4=(
                "ganho_vs_v4",
                "min"
            ),

            maior_ganho_vs_v4=(
                "ganho_vs_v4",
                "max"
            ),

            blocos_melhor_v4=(
                "ganho_vs_v4",
                lambda serie:
                    int(
                        (
                            serie > 0
                        )
                        .sum()
                    )
            ),

            blocos_pior_v4=(
                "ganho_vs_v4",
                lambda serie:
                    int(
                        (
                            serie < 0
                        )
                        .sum()
                    )
            ),
        )
    )

    resumo = (
        resumo
        .sort_values(
            "ganho_medio_vs_v4",
            ascending=False
        )
    )

    return (
        comparacao,
        resumo
    )


# ============================================================
# MOSTRAR
# ============================================================

def mostrar_resultados(
    matriz_lift,
    comparacao_v4
):

    principais = (
        filtrar_principais(
            matriz_lift
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "ROBUSTEZ TEMPORAL - PRINCIPAIS MODELOS"
    )

    print(
        "=" * 120
    )

    print(
        principais
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
        "GANHO DOS MODELOS CONTRA V4"
    )

    print(
        "=" * 120
    )

    print(
        comparacao_v4
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# EXPORTAR
# ============================================================

def exportar(
    resumo_blocos,
    score_blocos,
    matriz_lift,
    matriz_4,
    matriz_5,
    matriz_6,
    matriz_7,
    comparacao_detalhe,
    comparacao_resumo,
):

    with pd.ExcelWriter(
        ARQUIVO_SAIDA,
        engine="openpyxl"
    ) as writer:

        matriz_lift.to_excel(
            writer,
            sheet_name="Robustez_Modelos",
            index=False
        )

        comparacao_resumo.to_excel(
            writer,
            sheet_name="Vs_V4",
            index=False
        )

        score_blocos.to_excel(
            writer,
            sheet_name="Score_Blocos",
            index=False
        )

        resumo_blocos.to_excel(
            writer,
            sheet_name="Bloco_Exclusao",
            index=False
        )

        matriz_4.to_excel(
            writer,
            sheet_name="Top4",
            index=False
        )

        matriz_5.to_excel(
            writer,
            sheet_name="Top5",
            index=False
        )

        matriz_6.to_excel(
            writer,
            sheet_name="Top6",
            index=False
        )

        matriz_7.to_excel(
            writer,
            sheet_name="Top7",
            index=False
        )

        comparacao_detalhe.to_excel(
            writer,
            sheet_name="Vs_V4_Detalhes",
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

    detalhes = (
        carregar_dados()
    )

    resumo_blocos = (
        gerar_resumo_blocos(
            detalhes
        )
    )

    score_blocos = (
        gerar_score_por_bloco(
            resumo_blocos
        )
    )

    matriz_lift = (
        gerar_matriz_lift(
            score_blocos
        )
    )

    matriz_4 = (
        gerar_matriz_por_exclusao(
            resumo_blocos,
            4
        )
    )

    matriz_5 = (
        gerar_matriz_por_exclusao(
            resumo_blocos,
            5
        )
    )

    matriz_6 = (
        gerar_matriz_por_exclusao(
            resumo_blocos,
            6
        )
    )

    matriz_7 = (
        gerar_matriz_por_exclusao(
            resumo_blocos,
            7
        )
    )

    (
        comparacao_detalhe,
        comparacao_resumo
    ) = (
        gerar_comparacao_vs_v4(
            score_blocos
        )
    )

    mostrar_resultados(
        matriz_lift=
            matriz_lift,

        comparacao_v4=
            comparacao_resumo
    )

    exportar(
        resumo_blocos=
            resumo_blocos,

        score_blocos=
            score_blocos,

        matriz_lift=
            matriz_lift,

        matriz_4=
            matriz_4,

        matriz_5=
            matriz_5,

        matriz_6=
            matriz_6,

        matriz_7=
            matriz_7,

        comparacao_detalhe=
            comparacao_detalhe,

        comparacao_resumo=
            comparacao_resumo,
    )


if __name__ == "__main__":
    main()