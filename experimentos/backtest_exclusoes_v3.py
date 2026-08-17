import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# IMPORTS DA RAIZ
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from dados import carregar_resultados
from features_v2_reference import GeradorFeaturesV2

from ranking_v3 import (
    selecionar_exclusoes_v3
)


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


# Quantidade de dezenas consideradas
# "prováveis de sair".
QTD_PROVAVEIS = 5


ARQUIVO_SAIDA = (
    ROOT
    / "experimentos"
    / "resultado_backtest_exclusoes_v3.xlsx"
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
# AUXILIARES
# ============================================================

def dezenas_para_texto(
    dezenas
):

    return " ".join(
        f"{d:02d}"
        for d in sorted(dezenas)
    )


def esperado_aleatorio(
    qtd
):

    return (
        qtd
        * (10 / 25)
    )


# ============================================================
# FEATURES POR DEZENA
# ============================================================

def criar_features_por_dezena(
    X_teste,
    dezenas_teste
):
    """
    Converte:

        25 linhas × N features

    para:

        {
            1: {
                "freq_5": ...,
                ...
            },
            ...
        }
    """

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

def criar_ranking(
    modelo,
    X_teste,
    dezenas_teste
):
    """
    Retorna:

        ranking por probabilidade
        de NÃO sair.

    Primeiro item:
        maior P(não sair)
    """

    probabilidades = (
        modelo.predict_proba(
            X_teste
        )
    )

    classes = (
        modelo.classes_
    )

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

    ranking = []

    for dezena, prob in zip(
        dezenas_teste,
        prob_sair
    ):

        prob = float(
            prob
        )

        ranking.append({
            "dezena":
                int(dezena),

            "prob_sair":
                prob,

            "prob_nao_sair":
                1.0 - prob
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
# PROVÁVEIS DE SAIR
# ============================================================

def obter_provaveis(
    ranking,
    quantidade
):
    """
    Usa o próprio ranking do V2.

    Pega as dezenas com MAIOR
    probabilidade de sair.
    """

    ordenadas = sorted(
        ranking,
        key=lambda item:
            item[
                "prob_sair"
            ],
        reverse=True
    )

    return {
        item["dezena"]:
            item["prob_sair"]

        for item
        in ordenadas[
            :quantidade
        ]
    }


# ============================================================
# DATASET
# ============================================================

def preparar_dataset():

    print("=" * 90)
    print(
        "BACKTEST V3 - "
        "EXCLUSÃO + PROVÁVEIS + EXAUSTÃO"
    )
    print("=" * 90)

    print()
    print(
        "Carregando histórico..."
    )

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
        "Criando GeradorFeaturesV2 "
        "de referência..."
    )

    gerador = (
        GeradorFeaturesV2(
            df_bolas
        )
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
    ) = (
        gerador
        .construir_dataset(
            janela_minima=
                JANELA_MINIMA
        )
    )

    print(
        f"Dataset criado em "
        f"{time.time() - inicio:.1f}s"
    )

    print(
        f"X: {X.shape}"
    )

    print(
        f"y: {y.shape}"
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

    total = (
        gerador
        .total_sorteios
    )

    inicio_backtest = max(
        JANELA_MINIMA + 1,
        total - ULTIMOS_CONCURSOS
    )

    quantidade_testes = (
        total
        - inicio_backtest
    )

    print()
    print(
        f"Executando V3 nos últimos "
        f"{quantidade_testes} concursos..."
    )

    resultados = []

    inicio_total = time.time()

    # ========================================================
    # WALK-FORWARD
    # ========================================================

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
        # ====================================================

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

        # ====================================================
        # TESTE
        # ====================================================

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

            print(
                f"AVISO: "
                f"índice {indice_alvo} "
                f"possui "
                f"{len(X_teste)} linhas."
            )

            continue

        # ====================================================
        # TREINAR MODELO
        # ====================================================

        modelo = (
            criar_modelo()
        )

        modelo.fit(
            X_treino,
            y_treino
        )

        # ====================================================
        # RANKING V2
        # ====================================================

        ranking = (
            criar_ranking(
                modelo,
                X_teste,
                dezenas_teste
            )
        )

        # ====================================================
        # PROVÁVEIS
        # ====================================================

        provaveis = (
            obter_provaveis(
                ranking,
                QTD_PROVAVEIS
            )
        )

        # ====================================================
        # FEATURES POR DEZENA
        # ====================================================

        features_por_dezena = (
            criar_features_por_dezena(
                X_teste,
                dezenas_teste
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

        # ====================================================
        # CENÁRIOS
        # ====================================================

        for qtd in (
            CENARIOS_EXCLUSOES
        ):

            # -----------------------------------------------
            # V2 PURO
            # -----------------------------------------------

            exclusoes_v2 = {
                item[
                    "dezena"
                ]

                for item
                in ranking[
                    :qtd
                ]
            }

            acertos_v2 = len(
                exclusoes_v2
                & nao_sorteadas
            )

            # -----------------------------------------------
            # V3
            # -----------------------------------------------

            (
                exclusoes_v3_lista,
                detalhes_v3
            ) = (
                selecionar_exclusoes_v3(
                    ranking_exclusao=
                        ranking,

                    provaveis=
                        provaveis,

                    features_por_dezena=
                        features_por_dezena,

                    quantidade_exclusoes=
                        qtd
                )
            )

            exclusoes_v3 = set(
                exclusoes_v3_lista
            )

            acertos_v3 = len(
                exclusoes_v3
                & nao_sorteadas
            )

            erros_v3 = (
                exclusoes_v3
                & sorteadas
            )

            # -----------------------------------------------
            # Estatísticas das decisões V3
            # -----------------------------------------------

            qtd_exaustao = sum(
                1
                for detalhe
                in detalhes_v3
                if detalhe.get(
                    "motivo"
                )
                == "EXAUSTAO_FORTE"
            )

            qtd_protegidas = sum(
                1
                for detalhe
                in detalhes_v3
                if detalhe.get(
                    "decisao"
                )
                == "PROTEGER"
            )

            qtd_top_soberano = sum(
                1
                for detalhe
                in detalhes_v3
                if detalhe.get(
                    "motivo"
                )
                == "TOP_EXCLUSAO_SOBERANO"
            )

            # -----------------------------------------------
            # Prováveis que realmente saíram
            # -----------------------------------------------

            set_provaveis = set(
                provaveis.keys()
            )

            acertos_provaveis = len(
                set_provaveis
                & sorteadas
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

                # ================================
                # V2
                # ================================

                "acertos_v2":
                    acertos_v2,

                "exclusoes_v2":
                    dezenas_para_texto(
                        exclusoes_v2
                    ),

                # ================================
                # V3
                # ================================

                "acertos_v3":
                    acertos_v3,

                "ganho_v3_vs_v2":
                    (
                        acertos_v3
                        - acertos_v2
                    ),

                "exclusoes_v3":
                    dezenas_para_texto(
                        exclusoes_v3
                    ),

                "erros_exclusao_v3":
                    len(
                        erros_v3
                    ),

                "sorteadas_preservadas_v3":
                    (
                        15
                        - len(
                            erros_v3
                        )
                    ),

                "exclusao_perfeita_v3":
                    int(
                        acertos_v3
                        == qtd
                    ),

                # ================================
                # REGRAS
                # ================================

                "qtd_exaustao":
                    qtd_exaustao,

                "qtd_protegidas":
                    qtd_protegidas,

                "qtd_top_soberano":
                    qtd_top_soberano,

                # ================================
                # PROVÁVEIS
                # ================================

                "provaveis":
                    dezenas_para_texto(
                        set_provaveis
                    ),

                "acertos_provaveis":
                    acertos_provaveis,

                "qtd_provaveis":
                    len(
                        set_provaveis
                    ),

                # ================================
                # RESULTADO REAL
                # ================================

                "sorteadas_reais":
                    dezenas_para_texto(
                        sorteadas
                    ),

                "nao_sorteadas_reais":
                    dezenas_para_texto(
                        nao_sorteadas
                    )
            })

        # ====================================================
        # PROGRESSO
        # ====================================================

        if (
            numero_teste == 1
            or numero_teste % 5 == 0
            or numero_teste
            == quantidade_testes
        ):

            tempo_teste = (
                time.time()
                - inicio_teste
            )

            total_decorrido = (
                time.time()
                - inicio_total
            )

            print(
                f"{numero_teste:03d}/"
                f"{quantidade_testes}"
                f" | último="
                f"{tempo_teste:.2f}s"
                f" | total="
                f"{total_decorrido:.1f}s"
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

        media_v2 = (
            dados[
                "acertos_v2"
            ]
            .mean()
        )

        media_v3 = (
            dados[
                "acertos_v3"
            ]
            .mean()
        )

        esperado = (
            esperado_aleatorio(
                qtd
            )
        )

        ganhou = int(
            (
                dados[
                    "ganho_v3_vs_v2"
                ]
                > 0
            )
            .sum()
        )

        perdeu = int(
            (
                dados[
                    "ganho_v3_vs_v2"
                ]
                < 0
            )
            .sum()
        )

        empatou = int(
            (
                dados[
                    "ganho_v3_vs_v2"
                ]
                == 0
            )
            .sum()
        )

        perfeitas = int(
            dados[
                "exclusao_perfeita_v3"
            ]
            .sum()
        )

        linhas.append({
            "qtd_exclusoes":
                qtd,

            "qtd_candidatas":
                25 - qtd,

            "concursos":
                len(dados),

            "media_v2":
                media_v2,

            "media_v3":
                media_v3,

            "aleatorio":
                esperado,

            "ganho_v3_vs_v2":
                (
                    media_v3
                    - media_v2
                ),

            "ganho_v3_vs_aleatorio":
                (
                    media_v3
                    - esperado
                ),

            "ganho_percentual_vs_aleatorio":
                (
                    (
                        media_v3
                        / esperado
                    )
                    - 1
                )
                * 100,

            "v3_melhor":
                ganhou,

            "v3_pior":
                perdeu,

            "v3_igual":
                empatou,

            "media_exaustoes_forcadas":
                dados[
                    "qtd_exaustao"
                ]
                .mean(),

            "media_protegidas":
                dados[
                    "qtd_protegidas"
                ]
                .mean(),

            "media_acertos_provaveis":
                dados[
                    "acertos_provaveis"
                ]
                .mean(),

            "exclusoes_perfeitas_v3":
                perfeitas,

            "media_sorteadas_preservadas_v3":
                dados[
                    "sorteadas_preservadas_v3"
                ]
                .mean()
        })

    return (
        pd.DataFrame(
            linhas
        )
    )


# ============================================================
# DISTRIBUIÇÃO
# ============================================================

def gerar_distribuicao(
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

        total = len(
            dados
        )

        for acertos in range(
            qtd + 1
        ):

            qtd_v2 = int(
                (
                    dados[
                        "acertos_v2"
                    ]
                    == acertos
                )
                .sum()
            )

            qtd_v3 = int(
                (
                    dados[
                        "acertos_v3"
                    ]
                    == acertos
                )
                .sum()
            )

            linhas.append({
                "qtd_exclusoes":
                    qtd,

                "acertos":
                    acertos,

                "concursos_v2":
                    qtd_v2,

                "percentual_v2":
                    (
                        qtd_v2
                        / total
                        * 100
                    ),

                "concursos_v3":
                    qtd_v3,

                "percentual_v3":
                    (
                        qtd_v3
                        / total
                        * 100
                    )
            })

    return (
        pd.DataFrame(
            linhas
        )
    )


# ============================================================
# MOSTRAR RESUMO
# ============================================================

def mostrar_resumo(
    resumo
):

    print()
    print("=" * 120)
    print(
        "RESULTADO V3 - "
        "V2 PURO VS V3 HÍBRIDO"
    )
    print("=" * 120)

    colunas = [
        "qtd_exclusoes",
        "qtd_candidatas",

        "media_v2",
        "media_v3",

        "aleatorio",

        "ganho_v3_vs_v2",
        "ganho_percentual_vs_aleatorio",

        "v3_melhor",
        "v3_pior",
        "v3_igual",

        "media_exaustoes_forcadas",
        "media_protegidas",

        "media_acertos_provaveis",

        "exclusoes_perfeitas_v3",

        "media_sorteadas_preservadas_v3"
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