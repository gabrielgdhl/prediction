import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# IMPORTS DA RAIZ
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dados import carregar_resultados


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_SAIDA = (
    ROOT
    / "analises"
    / "estatisticas_repeticoes.xlsx"
)

PROBABILIDADE_BASE = 15 / 25  # 60%

MAX_ATRASO_ANALISADO = 10
MAX_SEQUENCIA_ANALISADA = 10


# ============================================================
# MATRIZ BINÁRIA
# ============================================================

def criar_matriz_binaria(df_bolas):
    matriz = np.zeros(
        (len(df_bolas), 25),
        dtype=np.int8
    )

    for indice, jogo in enumerate(df_bolas.to_numpy()):
        for dezena in jogo:
            matriz[
                indice,
                int(dezena) - 1
            ] = 1

    return matriz


# ============================================================
# SEQUÊNCIAS
# ============================================================

def extrair_sequencias(serie, valor):
    sequencias = []
    atual = 0

    for item in serie:
        if item == valor:
            atual += 1
        else:
            if atual > 0:
                sequencias.append(atual)

            atual = 0

    if atual > 0:
        sequencias.append(atual)

    return sequencias


# ============================================================
# ESTADO ATUAL
# ============================================================

def calcular_atraso_atual(serie):
    atraso = 0

    for valor in reversed(serie):
        if valor == 1:
            break

        atraso += 1

    return atraso


def calcular_sequencia_presenca_atual(serie):
    sequencia = 0

    for valor in reversed(serie):
        if valor == 0:
            break

        sequencia += 1

    return sequencia


# ============================================================
# INTERVALO DE CONFIANÇA
# ============================================================

def intervalo_confianca_wilson(
    sucessos,
    total,
    z=1.96
):
    """
    Intervalo de confiança de Wilson para proporções.

    Retorna:
        limite_inferior
        limite_superior
    """

    if total == 0:
        return np.nan, np.nan

    p = sucessos / total

    denominador = 1 + (z ** 2 / total)

    centro = (
        p
        + (z ** 2 / (2 * total))
    )

    margem = z * np.sqrt(
        (
            p * (1 - p)
            + (z ** 2 / (4 * total))
        )
        / total
    )

    inferior = (
        (centro - margem)
        / denominador
    )

    superior = (
        (centro + margem)
        / denominador
    )

    return (
        max(0.0, inferior),
        min(1.0, superior)
    )


# ============================================================
# FORÇA ESTATÍSTICA
# ============================================================

def calcular_forca_estatistica(
    probabilidade,
    amostras,
    limite_inferior,
    limite_superior
):
    """
    Classificação inicial.

    Não é usada pelo ML ainda.

    Serve para interpretar se o padrão
    merece atenção.
    """

    if (
        np.isnan(probabilidade)
        or amostras == 0
    ):
        return "SEM_DADOS"

    lift = (
        probabilidade
        - PROBABILIDADE_BASE
    )

    if amostras < 20:
        return "AMOSTRA_BAIXA"

    if (
        limite_inferior
        > PROBABILIDADE_BASE
    ):
        if lift >= 0.08:
            return "MUITO_FORTE_POSITIVO"

        if lift >= 0.03:
            return "FORTE_POSITIVO"

        return "POSITIVO"

    if (
        limite_superior
        < PROBABILIDADE_BASE
    ):
        if lift <= -0.08:
            return "MUITO_FORTE_NEGATIVO"

        if lift <= -0.03:
            return "FORTE_NEGATIVO"

        return "NEGATIVO"

    if abs(lift) >= 0.05:
        return "SINAL_SEM_CONFIRMACAO"

    return "NEUTRO"


# ============================================================
# ATRASO
# ============================================================

def estatistica_sair_apos_atraso(
    serie,
    atraso_desejado
):
    total_situacoes = 0
    saiu_depois = 0

    atraso_atual = 0

    for indice in range(
        len(serie) - 1
    ):
        if serie[indice] == 1:
            atraso_atual = 0
        else:
            atraso_atual += 1

        if atraso_atual == atraso_desejado:
            total_situacoes += 1

            if serie[indice + 1] == 1:
                saiu_depois += 1

    if total_situacoes == 0:
        return {
            "probabilidade": np.nan,
            "amostras": 0,
            "sucessos": 0,
            "ic_inferior": np.nan,
            "ic_superior": np.nan,
        }

    probabilidade = (
        saiu_depois
        / total_situacoes
    )

    inferior, superior = (
        intervalo_confianca_wilson(
            saiu_depois,
            total_situacoes
        )
    )

    return {
        "probabilidade":
            probabilidade,

        "amostras":
            total_situacoes,

        "sucessos":
            saiu_depois,

        "ic_inferior":
            inferior,

        "ic_superior":
            superior,
    }


# ============================================================
# REPETIÇÃO
# ============================================================

def estatistica_repetir_apos_sequencia(
    serie,
    tamanho_sequencia
):
    total_situacoes = 0
    repetiu = 0

    sequencia_atual = 0

    for indice in range(
        len(serie) - 1
    ):
        if serie[indice] == 1:
            sequencia_atual += 1
        else:
            sequencia_atual = 0

        if (
            sequencia_atual
            == tamanho_sequencia
        ):
            total_situacoes += 1

            if serie[indice + 1] == 1:
                repetiu += 1

    if total_situacoes == 0:
        return {
            "probabilidade": np.nan,
            "amostras": 0,
            "sucessos": 0,
            "ic_inferior": np.nan,
            "ic_superior": np.nan,
        }

    probabilidade = (
        repetiu
        / total_situacoes
    )

    inferior, superior = (
        intervalo_confianca_wilson(
            repetiu,
            total_situacoes
        )
    )

    return {
        "probabilidade":
            probabilidade,

        "amostras":
            total_situacoes,

        "sucessos":
            repetiu,

        "ic_inferior":
            inferior,

        "ic_superior":
            superior,
    }


# ============================================================
# PERCENTIL DO ATRASO
# ============================================================

def calcular_percentil_atraso(
    atraso_atual,
    sequencias_ausencia
):
    if not sequencias_ausencia:
        return 0.0

    quantidade = sum(
        atraso <= atraso_atual
        for atraso
        in sequencias_ausencia
    )

    return (
        quantidade
        / len(sequencias_ausencia)
    )


# ============================================================
# ANALISAR DEZENA
# ============================================================

def analisar_dezena(
    numero,
    serie
):
    frequencia_historica = float(
        np.mean(serie)
    )

    sequencias_ausencia = (
        extrair_sequencias(
            serie,
            0
        )
    )

    sequencias_presenca = (
        extrair_sequencias(
            serie,
            1
        )
    )

    atraso_atual = (
        calcular_atraso_atual(
            serie
        )
    )

    sequencia_atual = (
        calcular_sequencia_presenca_atual(
            serie
        )
    )

    percentil_atraso = (
        calcular_percentil_atraso(
            atraso_atual,
            sequencias_ausencia
        )
    )

    # ========================================================
    # PADRÃO ATUAL
    # ========================================================

    if sequencia_atual > 0:
        tipo_padrao = "PRESENCA"

        stats_padrao = (
            estatistica_repetir_apos_sequencia(
                serie,
                sequencia_atual
            )
        )

    else:
        tipo_padrao = "AUSENCIA"

        stats_padrao = (
            estatistica_sair_apos_atraso(
                serie,
                atraso_atual
            )
        )

    prob_padrao = (
        stats_padrao[
            "probabilidade"
        ]
    )

    amostras = (
        stats_padrao[
            "amostras"
        ]
    )

    ic_inferior = (
        stats_padrao[
            "ic_inferior"
        ]
    )

    ic_superior = (
        stats_padrao[
            "ic_superior"
        ]
    )

    lift = (
        prob_padrao
        - PROBABILIDADE_BASE
        if not np.isnan(prob_padrao)
        else np.nan
    )

    lift_percentual = (
        (
            prob_padrao
            / PROBABILIDADE_BASE
        ) - 1
        if not np.isnan(prob_padrao)
        else np.nan
    )

    forca = (
        calcular_forca_estatistica(
            prob_padrao,
            amostras,
            ic_inferior,
            ic_superior
        )
    )

    # ========================================================
    # ESTATÍSTICAS DE AUSÊNCIA
    # ========================================================

    media_ausencia = (
        float(
            np.mean(
                sequencias_ausencia
            )
        )
        if sequencias_ausencia
        else 0.0
    )

    mediana_ausencia = (
        float(
            np.median(
                sequencias_ausencia
            )
        )
        if sequencias_ausencia
        else 0.0
    )

    max_ausencia = (
        max(
            sequencias_ausencia
        )
        if sequencias_ausencia
        else 0
    )

    # ========================================================
    # PRESENÇA
    # ========================================================

    media_presenca = (
        float(
            np.mean(
                sequencias_presenca
            )
        )
        if sequencias_presenca
        else 0.0
    )

    max_presenca = (
        max(
            sequencias_presenca
        )
        if sequencias_presenca
        else 0
    )

    return {
        "dezena":
            numero,

        "frequencia_historica":
            frequencia_historica,

        "tipo_padrao_atual":
            tipo_padrao,

        "atraso_atual":
            atraso_atual,

        "percentil_atraso":
            percentil_atraso,

        "sequencia_presenca_atual":
            sequencia_atual,

        "prob_sair_padrao_atual":
            prob_padrao,

        "probabilidade_base":
            PROBABILIDADE_BASE,

        "lift_pontos_percentuais":
            lift,

        "lift_percentual":
            lift_percentual,

        "amostras_padrao":
            amostras,

        "sucessos_padrao":
            stats_padrao[
                "sucessos"
            ],

        "ic_95_inferior":
            ic_inferior,

        "ic_95_superior":
            ic_superior,

        "forca_estatistica":
            forca,

        "media_ausencia":
            media_ausencia,

        "mediana_ausencia":
            mediana_ausencia,

        "max_ausencia":
            max_ausencia,

        "media_presenca":
            media_presenca,

        "max_presenca":
            max_presenca,
    }


# ============================================================
# TABELA DE ATRASOS
# ============================================================

def gerar_tabela_atrasos(
    numero,
    serie
):
    linhas = []

    for atraso in range(
        0,
        MAX_ATRASO_ANALISADO + 1
    ):
        stats = (
            estatistica_sair_apos_atraso(
                serie,
                atraso
            )
        )

        prob = (
            stats[
                "probabilidade"
            ]
        )

        lift = (
            prob
            - PROBABILIDADE_BASE
            if not np.isnan(prob)
            else np.nan
        )

        linhas.append({
            "dezena":
                numero,

            "atraso":
                atraso,

            "prob_sair_proximo":
                prob,

            "probabilidade_base":
                PROBABILIDADE_BASE,

            "lift_pontos_percentuais":
                lift,

            "amostras":
                stats["amostras"],

            "sucessos":
                stats["sucessos"],

            "ic_95_inferior":
                stats["ic_inferior"],

            "ic_95_superior":
                stats["ic_superior"],

            "forca_estatistica":
                calcular_forca_estatistica(
                    prob,
                    stats["amostras"],
                    stats["ic_inferior"],
                    stats["ic_superior"]
                )
        })

    return linhas


# ============================================================
# TABELA DE REPETIÇÃO
# ============================================================

def gerar_tabela_repeticoes(
    numero,
    serie
):
    linhas = []

    for sequencia in range(
        1,
        MAX_SEQUENCIA_ANALISADA + 1
    ):
        stats = (
            estatistica_repetir_apos_sequencia(
                serie,
                sequencia
            )
        )

        prob = (
            stats[
                "probabilidade"
            ]
        )

        lift = (
            prob
            - PROBABILIDADE_BASE
            if not np.isnan(prob)
            else np.nan
        )

        linhas.append({
            "dezena":
                numero,

            "sequencia_presenca":
                sequencia,

            "prob_repetir":
                prob,

            "probabilidade_base":
                PROBABILIDADE_BASE,

            "lift_pontos_percentuais":
                lift,

            "amostras":
                stats["amostras"],

            "sucessos":
                stats["sucessos"],

            "ic_95_inferior":
                stats["ic_inferior"],

            "ic_95_superior":
                stats["ic_superior"],

            "forca_estatistica":
                calcular_forca_estatistica(
                    prob,
                    stats["amostras"],
                    stats["ic_inferior"],
                    stats["ic_superior"]
                )
        })

    return linhas


# ============================================================
# SCORE DE CONFIANÇA
# ============================================================

def calcular_score_confianca(
    linha
):
    """
    Score somente exploratório.

    Ainda NÃO deve ser utilizado
    diretamente para apostar.

    Considera:

        probabilidade do padrão
        +
        tamanho da amostra
        +
        distância da base de 60%
    """

    prob = (
        linha[
            "prob_sair_padrao_atual"
        ]
    )

    amostras = (
        linha[
            "amostras_padrao"
        ]
    )

    if np.isnan(prob):
        return 0.0

    # Confiança na quantidade de dados.
    #
    # Vai se aproximando de 1 conforme
    # aumenta a amostra.
    confianca_amostra = (
        amostras
        / (
            amostras + 50
        )
    )

    lift = (
        prob
        - PROBABILIDADE_BASE
    )

    score = (
        PROBABILIDADE_BASE
        +
        (
            lift
            * confianca_amostra
        )
    )

    return float(score)


# ============================================================
# CLASSIFICAÇÃO DE PEDRAS
# ============================================================

def classificar_pedra(
    linha
):
    """
    Classificação exploratória.

    Posteriormente será validada no backtest.
    """

    forca = (
        linha[
            "forca_estatistica"
        ]
    )

    score = (
        linha[
            "score_confianca"
        ]
    )

    amostras = (
        linha[
            "amostras_padrao"
        ]
    )

    if (
        forca
        in {
            "MUITO_FORTE_POSITIVO",
            "FORTE_POSITIVO"
        }
        and score >= 0.64
        and amostras >= 30
    ):
        return "CANDIDATA_OBRIGATORIA"

    if (
        score >= 0.61
        and amostras >= 20
    ):
        return "PREFERENCIAL"

    if score < 0.57:
        return "RISCO_EXCLUSAO"

    return "NEUTRA"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print(
        "ANÁLISE ESTATÍSTICA DE "
        "REPETIÇÃO E AUSÊNCIA"
    )
    print("=" * 80)

    _, df_bolas = carregar_resultados(
        ROOT
        / "lotofacil_resultados.xlsx"
    )

    matriz = criar_matriz_binaria(
        df_bolas
    )

    resumo = []
    atrasos = []
    repeticoes = []

    for indice_dezena in range(25):
        numero = indice_dezena + 1

        serie = matriz[
            :,
            indice_dezena
        ]

        resumo.append(
            analisar_dezena(
                numero,
                serie
            )
        )

        atrasos.extend(
            gerar_tabela_atrasos(
                numero,
                serie
            )
        )

        repeticoes.extend(
            gerar_tabela_repeticoes(
                numero,
                serie
            )
        )

    df_resumo = pd.DataFrame(
        resumo
    )

    df_atrasos = pd.DataFrame(
        atrasos
    )

    df_repeticoes = pd.DataFrame(
        repeticoes
    )

    # ========================================================
    # SCORE EXPLORATÓRIO
    # ========================================================

    df_resumo[
        "score_confianca"
    ] = (
        df_resumo.apply(
            calcular_score_confianca,
            axis=1
        )
    )

    df_resumo[
        "classificacao"
    ] = (
        df_resumo.apply(
            classificar_pedra,
            axis=1
        )
    )

    df_resumo = (
        df_resumo
        .sort_values(
            [
                "score_confianca",
                "amostras_padrao"
            ],
            ascending=[
                False,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df_resumo[
        "ranking"
    ] = (
        np.arange(
            1,
            len(df_resumo) + 1
        )
    )

    # ========================================================
    # EXPORTAR
    # ========================================================

    ARQUIVO_SAIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with pd.ExcelWriter(
        ARQUIVO_SAIDA,
        engine="openpyxl"
    ) as writer:
        df_resumo.to_excel(
            writer,
            sheet_name="Resumo",
            index=False
        )

        df_atrasos.to_excel(
            writer,
            sheet_name="Prob_por_Atraso",
            index=False
        )

        df_repeticoes.to_excel(
            writer,
            sheet_name="Prob_Repeticao",
            index=False
        )

    # ========================================================
    # TERMINAL
    # ========================================================

    colunas = [
        "ranking",
        "dezena",
        "tipo_padrao_atual",
        "atraso_atual",
        "sequencia_presenca_atual",
        "prob_sair_padrao_atual",
        "lift_pontos_percentuais",
        "amostras_padrao",
        "ic_95_inferior",
        "ic_95_superior",
        "forca_estatistica",
        "score_confianca",
        "classificacao"
    ]

    print()
    print(
        df_resumo[
            colunas
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Arquivo gerado: "
        f"{ARQUIVO_SAIDA}"
    )


if __name__ == "__main__":
    main()