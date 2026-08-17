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

MAX_SEQUENCIA = 12

ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "diagnostico_repeticao.xlsx"
)


# ============================================================
# MATRIZ BINÁRIA
# ============================================================

def criar_matriz_binaria(df_bolas):
    """
    Linhas:
        concursos

    Colunas:
        dezenas 01..25

    Valor:
        1 = saiu
        0 = não saiu
    """

    matriz = np.zeros(
        (
            len(df_bolas),
            25
        ),
        dtype=np.int8
    )

    for indice, sorteio in enumerate(
        df_bolas.to_numpy()
    ):
        for dezena in sorteio:
            dezena = int(dezena)

            if not 1 <= dezena <= 25:
                raise ValueError(
                    f"Dezena inválida: {dezena}"
                )

            matriz[
                indice,
                dezena - 1
            ] = 1

    return matriz


# ============================================================
# INTERVALO DE CONFIANÇA - WILSON
# ============================================================

def intervalo_wilson(
    sucessos,
    total,
    z=1.96
):
    """
    Intervalo de confiança para proporção.

    Útil para não confiar demais em:

        90% com 10 amostras

    comparado com:

        90% com 500 amostras
    """

    if total == 0:
        return np.nan, np.nan

    p = sucessos / total

    denominador = (
        1
        + (z ** 2 / total)
    )

    centro = (
        p
        + (
            z ** 2
            / (2 * total)
        )
    )

    margem = (
        z
        * np.sqrt(
            (
                p * (1 - p)
                + (
                    z ** 2
                    / (4 * total)
                )
            )
            / total
        )
    )

    inferior = (
        centro - margem
    ) / denominador

    superior = (
        centro + margem
    ) / denominador

    return (
        max(0.0, inferior),
        min(1.0, superior)
    )


# ============================================================
# ANALISAR SEQUÊNCIAS DE UMA DEZENA
# ============================================================

def analisar_sequencias_dezena(
    serie,
    numero_dezena
):
    """
    Para cada estado:

        saiu 1 vez seguida
        saiu 2 vezes seguidas
        saiu 3 vezes seguidas
        ...

    verificamos o próximo concurso.

    Exemplo:

        sequência = 3

    pergunta:

        depois de chegar a 3 saídas seguidas,
        saiu novamente?

    Isso produz:

        P(repetir | sequência=3)

    e:

        P(exaustão | sequência=3)
        =
        1 - P(repetir)
    """

    resultados = []

    for sequencia_desejada in range(
        1,
        MAX_SEQUENCIA + 1
    ):
        amostras = 0
        repetiu = 0
        parou = 0

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
                == sequencia_desejada
            ):
                amostras += 1

                if (
                    serie[indice + 1]
                    == 1
                ):
                    repetiu += 1
                else:
                    parou += 1

        if amostras > 0:
            prob_repetir = (
                repetiu
                / amostras
            )

            prob_exaustao = (
                parou
                / amostras
            )

            (
                ic_repetir_inf,
                ic_repetir_sup
            ) = intervalo_wilson(
                repetiu,
                amostras
            )

            (
                ic_exaustao_inf,
                ic_exaustao_sup
            ) = intervalo_wilson(
                parou,
                amostras
            )

        else:
            prob_repetir = np.nan
            prob_exaustao = np.nan

            ic_repetir_inf = np.nan
            ic_repetir_sup = np.nan

            ic_exaustao_inf = np.nan
            ic_exaustao_sup = np.nan

        resultados.append({
            "dezena":
                numero_dezena,

            "sequencia":
                sequencia_desejada,

            "amostras":
                amostras,

            "repetiu":
                repetiu,

            "parou":
                parou,

            "prob_repetir":
                prob_repetir,

            "prob_exaustao":
                prob_exaustao,

            "ic95_repetir_inferior":
                ic_repetir_inf,

            "ic95_repetir_superior":
                ic_repetir_sup,

            "ic95_exaustao_inferior":
                ic_exaustao_inf,

            "ic95_exaustao_superior":
                ic_exaustao_sup,
        })

    return resultados


# ============================================================
# ESTADO ATUAL
# ============================================================

def calcular_sequencia_atual(
    serie
):
    """
    Quantas vezes consecutivas a dezena
    está saindo neste momento.
    """

    sequencia = 0

    for valor in reversed(
        serie
    ):
        if valor == 0:
            break

        sequencia += 1

    return sequencia


# ============================================================
# CLASSIFICAR FORÇA DE EXAUSTÃO
# ============================================================

def classificar_exaustao(
    prob_exaustao,
    amostras,
    ic_inferior
):
    """
    Classificação apenas diagnóstica.

    Não entra ainda no ranking V3.
    """

    if (
        np.isnan(prob_exaustao)
        or amostras == 0
    ):
        return "SEM_DADOS"

    if amostras < 20:
        return "AMOSTRA_BAIXA"

    # Mais conservador:
    # exige que até o limite inferior
    # do IC seja alto.
    if (
        prob_exaustao >= 0.85
        and ic_inferior >= 0.70
    ):
        return "EXAUSTAO_MUITO_FORTE"

    if (
        prob_exaustao >= 0.75
        and ic_inferior >= 0.60
    ):
        return "EXAUSTAO_FORTE"

    if (
        prob_exaustao >= 0.65
    ):
        return "EXAUSTAO_MODERADA"

    return "SEM_SINAL_FORTE"


# ============================================================
# RESUMO DA SITUAÇÃO ATUAL
# ============================================================

def gerar_resumo_atual(
    matriz,
    df_detalhes
):
    linhas = []

    for indice_dezena in range(
        25
    ):
        dezena = (
            indice_dezena + 1
        )

        serie = (
            matriz[
                :,
                indice_dezena
            ]
        )

        sequencia_atual = (
            calcular_sequencia_atual(
                serie
            )
        )

        if sequencia_atual <= 0:
            linhas.append({
                "dezena":
                    dezena,

                "sequencia_atual":
                    0,

                "amostras":
                    0,

                "prob_repetir":
                    np.nan,

                "prob_exaustao":
                    np.nan,

                "ic95_exaustao_inferior":
                    np.nan,

                "ic95_exaustao_superior":
                    np.nan,

                "classificacao":
                    "NAO_ESTA_EM_SEQUENCIA"
            })

            continue

        dados = (
            df_detalhes[
                (
                    df_detalhes[
                        "dezena"
                    ]
                    == dezena
                )
                &
                (
                    df_detalhes[
                        "sequencia"
                    ]
                    == sequencia_atual
                )
            ]
        )

        if len(dados) == 0:
            linhas.append({
                "dezena":
                    dezena,

                "sequencia_atual":
                    sequencia_atual,

                "amostras":
                    0,

                "prob_repetir":
                    np.nan,

                "prob_exaustao":
                    np.nan,

                "ic95_exaustao_inferior":
                    np.nan,

                "ic95_exaustao_superior":
                    np.nan,

                "classificacao":
                    "SEM_DADOS"
            })

            continue

        linha = (
            dados.iloc[0]
        )

        classificacao = (
            classificar_exaustao(
                linha[
                    "prob_exaustao"
                ],
                int(
                    linha[
                        "amostras"
                    ]
                ),
                linha[
                    "ic95_exaustao_inferior"
                ]
            )
        )

        linhas.append({
            "dezena":
                dezena,

            "sequencia_atual":
                sequencia_atual,

            "amostras":
                int(
                    linha[
                        "amostras"
                    ]
                ),

            "prob_repetir":
                linha[
                    "prob_repetir"
                ],

            "prob_exaustao":
                linha[
                    "prob_exaustao"
                ],

            "ic95_exaustao_inferior":
                linha[
                    "ic95_exaustao_inferior"
                ],

            "ic95_exaustao_superior":
                linha[
                    "ic95_exaustao_superior"
                ],

            "classificacao":
                classificacao
        })

    return pd.DataFrame(
        linhas
    )


# ============================================================
# TOP SINAIS HISTÓRICOS
# ============================================================

def gerar_top_exaustao(
    df_detalhes,
    minimo_amostras=20
):
    """
    Mostra as combinações:

        dezena + sequência

    que historicamente apresentam
    maior probabilidade de parar.
    """

    dados = (
        df_detalhes[
            df_detalhes[
                "amostras"
            ]
            >= minimo_amostras
        ]
        .copy()
    )

    dados = (
        dados
        .sort_values(
            [
                "prob_exaustao",
                "amostras"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    return dados


# ============================================================
# RESUMO GLOBAL POR TAMANHO DA SEQUÊNCIA
# ============================================================

def gerar_resumo_global(
    matriz
):
    """
    Além da análise por dezena,
    mostra o comportamento geral da Lotofácil.

    Exemplo:

        sequência 1
        sequência 2
        sequência 3
        ...
    """

    linhas = []

    for sequencia_desejada in range(
        1,
        MAX_SEQUENCIA + 1
    ):
        amostras = 0
        repetiu = 0
        parou = 0

        for indice_dezena in range(
            25
        ):
            serie = (
                matriz[
                    :,
                    indice_dezena
                ]
            )

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
                    == sequencia_desejada
                ):
                    amostras += 1

                    if (
                        serie[
                            indice + 1
                        ]
                        == 1
                    ):
                        repetiu += 1
                    else:
                        parou += 1

        if amostras > 0:
            prob_repetir = (
                repetiu / amostras
            )

            prob_exaustao = (
                parou / amostras
            )
        else:
            prob_repetir = np.nan
            prob_exaustao = np.nan

        linhas.append({
            "sequencia":
                sequencia_desejada,

            "amostras":
                amostras,

            "repetiu":
                repetiu,

            "parou":
                parou,

            "prob_repetir":
                prob_repetir,

            "prob_exaustao":
                prob_exaustao
        })

    return pd.DataFrame(
        linhas
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "=" * 85
    )

    print(
        "DIAGNÓSTICO DE REPETIÇÃO / "
        "EXAUSTÃO - LOTOFÁCIL"
    )

    print(
        "=" * 85
    )

    _, df_bolas = (
        carregar_resultados(
            ROOT
            / "lotofacil_resultados.xlsx"
        )
    )

    matriz = (
        criar_matriz_binaria(
            df_bolas
        )
    )

    print()
    print(
        f"Concursos analisados: "
        f"{len(matriz)}"
    )

    # ========================================================
    # DETALHES POR DEZENA
    # ========================================================

    registros = []

    for indice_dezena in range(
        25
    ):
        dezena = (
            indice_dezena + 1
        )

        serie = (
            matriz[
                :,
                indice_dezena
            ]
        )

        registros.extend(
            analisar_sequencias_dezena(
                serie,
                dezena
            )
        )

    df_detalhes = pd.DataFrame(
        registros
    )

    # ========================================================
    # SITUAÇÃO ATUAL
    # ========================================================

    df_atual = (
        gerar_resumo_atual(
            matriz,
            df_detalhes
        )
    )

    # ========================================================
    # TOP EXAUSTÃO
    # ========================================================

    df_top = (
        gerar_top_exaustao(
            df_detalhes,
            minimo_amostras=20
        )
    )

    # ========================================================
    # GLOBAL
    # ========================================================

    df_global = (
        gerar_resumo_global(
            matriz
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

        df_atual.to_excel(
            writer,
            sheet_name="Situacao_Atual",
            index=False
        )

        df_top.to_excel(
            writer,
            sheet_name="Top_Exaustao",
            index=False
        )

        df_detalhes.to_excel(
            writer,
            sheet_name="Detalhes",
            index=False
        )

        df_global.to_excel(
            writer,
            sheet_name="Global",
            index=False
        )

    # ========================================================
    # TERMINAL
    # ========================================================

    print()
    print(
        "=" * 85
    )

    print(
        "SITUAÇÃO ATUAL"
    )

    print(
        "=" * 85
    )

    atuais_em_sequencia = (
        df_atual[
            df_atual[
                "sequencia_atual"
            ]
            > 0
        ]
        .sort_values(
            [
                "prob_exaustao",
                "amostras"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    print(
        atuais_em_sequencia
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 85
    )

    print(
        "COMPORTAMENTO GLOBAL"
    )

    print(
        "=" * 85
    )

    print(
        df_global
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Arquivo gerado:"
    )

    print(
        ARQUIVO_SAIDA
    )


if __name__ == "__main__":
    main()