import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ranking_v3 import (
    calcular_score_presenca,
    analisar_exaustao
)


# ============================================================
# FEATURES DO META-MODELO
# ============================================================

FEATURES_META = [
    # Modelo V2
    "prob_nao_sair_v2",
    "prob_sair_v2",

    # Presença estatística
    "score_presenca",
    "lift_presenca",
    "prob_condicional_presenca",
    "log_amostras_presenca",

    # Exaustão
    "prob_exaustao_ajustada",
    "lift_exaustao",
    "log_amostras_exaustao",

    # Frequências
    "freq_5",
    "freq_10",
    "freq_20",
    "freq_50",
    "freq_100",
    "freq_200",
    "freq_historica",

    # Estado
    "atraso_log",
    "percentil_atraso",
    "sequencia_presenca_log",
    "saiu_anterior",

    # Comparação com as outras dezenas
    "freq_20_relativa",
    "atraso_relativo",
    "ranking_freq_20",
    "ranking_atraso",
]


# ============================================================
# CONSTRUIR FEATURES DE UMA DEZENA
# ============================================================

def construir_features_meta(
    ranking_v2_item,
    features_dezena
):
    """
    Une:

        previsão do V2
        +
        estatísticas individuais

    em um vetor para o meta-modelo.

    Target futuro:

        1 = NÃO saiu
        0 = saiu
    """

    # ========================================================
    # PRESENÇA
    # ========================================================

    presenca = calcular_score_presenca(
        features_dezena
    )

    # ========================================================
    # EXAUSTÃO
    #
    # Aqui não interessa se passou por um threshold.
    # Queremos os valores contínuos para a Logistic Regression.
    # ========================================================

    exaustao = analisar_exaustao(
        features_dezena,

        # Valores baixos porque não usaremos
        # a decisão booleana.
        lift_minimo=-1.0,
        amostras_minimas=0
    )

    # ========================================================
    # TAMANHOS DE AMOSTRA
    # ========================================================

    amostras_presenca = (
        presenca[
            "amostras"
        ]
    )

    amostras_exaustao = (
        exaustao[
            "amostras"
        ]
    )

    linha = [
        # ----------------------------------------------------
        # V2
        # ----------------------------------------------------

        float(
            ranking_v2_item[
                "prob_nao_sair"
            ]
        ),

        float(
            ranking_v2_item[
                "prob_sair"
            ]
        ),

        # ----------------------------------------------------
        # PRESENÇA ESTATÍSTICA
        # ----------------------------------------------------

        float(
            presenca[
                "score_presenca"
            ]
        ),

        float(
            presenca[
                "lift_presenca"
            ]
        ),

        float(
            presenca[
                "prob_condicional"
            ]
        ),

        float(
            np.log1p(
                amostras_presenca
            )
        ),

        # ----------------------------------------------------
        # EXAUSTÃO
        # ----------------------------------------------------

        float(
            exaustao[
                "prob_exaustao_ajustada"
            ]
        ),

        float(
            exaustao[
                "lift_exaustao"
            ]
        ),

        float(
            np.log1p(
                amostras_exaustao
            )
        ),

        # ----------------------------------------------------
        # FREQUÊNCIAS
        # ----------------------------------------------------

        float(
            features_dezena[
                "freq_5"
            ]
        ),

        float(
            features_dezena[
                "freq_10"
            ]
        ),

        float(
            features_dezena[
                "freq_20"
            ]
        ),

        float(
            features_dezena[
                "freq_50"
            ]
        ),

        float(
            features_dezena[
                "freq_100"
            ]
        ),

        float(
            features_dezena[
                "freq_200"
            ]
        ),

        float(
            features_dezena[
                "freq_historica"
            ]
        ),

        # ----------------------------------------------------
        # ESTADO
        # ----------------------------------------------------

        float(
            np.log1p(
                max(
                    0.0,
                    features_dezena[
                        "atraso"
                    ]
                )
            )
        ),

        float(
            features_dezena[
                "percentil_atraso"
            ]
        ),

        float(
            np.log1p(
                max(
                    0.0,
                    features_dezena[
                        "sequencia_presenca"
                    ]
                )
            )
        ),

        float(
            features_dezena[
                "saiu_anterior"
            ]
        ),

        # ----------------------------------------------------
        # RELATIVAS
        # ----------------------------------------------------

        float(
            features_dezena[
                "freq_20_relativa"
            ]
        ),

        float(
            features_dezena[
                "atraso_relativo"
            ]
        ),

        float(
            features_dezena[
                "ranking_freq_20"
            ]
        ),

        float(
            features_dezena[
                "ranking_atraso"
            ]
        ),
    ]

    return np.asarray(
        linha,
        dtype=np.float64
    )


# ============================================================
# CRIAR MODELO V4
# ============================================================

def criar_modelo_v4():
    """
    StandardScaler:
        coloca as features em escalas comparáveis.

    LogisticRegression:
        aprende automaticamente os pesos.

    Não usamos class_weight='balanced' aqui porque
    queremos manter a distribuição real:

        ~40% não saem
        ~60% saem.
    """

    return Pipeline([
        (
            "scaler",

            StandardScaler()
        ),

        (
            "logistic",

            LogisticRegression(
                max_iter=3000,
                C=1.0,
                random_state=42
            )
        )
    ])


# ============================================================
# TREINAMENTO
# ============================================================

def treinar_v4(
    X_meta,
    y_meta
):
    modelo = (
        criar_modelo_v4()
    )

    modelo.fit(
        X_meta,
        y_meta
    )

    return modelo


# ============================================================
# PREDIÇÃO / RANKING
# ============================================================

def criar_ranking_v4(
    modelo,
    ranking_v2,
    features_por_dezena
):
    """
    Cria novo ranking das 25 dezenas.

    Classe:

        1 = NÃO sair

    Logo:

        maior P(classe 1)
        =
        melhor candidata à exclusão.
    """

    linhas = []
    dezenas = []

    ranking_v2_map = {
        item["dezena"]:
            item
        for item
        in ranking_v2
    }

    for dezena in range(
        1,
        26
    ):

        item_v2 = (
            ranking_v2_map[
                dezena
            ]
        )

        features = (
            features_por_dezena[
                dezena
            ]
        )

        linha = (
            construir_features_meta(
                item_v2,
                features
            )
        )

        linhas.append(
            linha
        )

        dezenas.append(
            dezena
        )

    X = np.asarray(
        linhas,
        dtype=np.float64
    )

    probabilidades = (
        modelo.predict_proba(
            X
        )
    )

    classes = (
        modelo.named_steps[
            "logistic"
        ].classes_
    )

    indice_nao_sair = (
        np.where(
            classes == 1
        )[0][0]
    )

    prob_nao_sair = (
        probabilidades[
            :,
            indice_nao_sair
        ]
    )

    ranking = []

    for dezena, prob in zip(
        dezenas,
        prob_nao_sair
    ):

        ranking.append({
            "dezena":
                dezena,

            "prob_nao_sair_v4":
                float(prob),

            "prob_sair_v4":
                float(
                    1.0 - prob
                )
        })

    ranking.sort(
        key=lambda item:
            item[
                "prob_nao_sair_v4"
            ],
        reverse=True
    )

    return ranking


# ============================================================
# PESOS APRENDIDOS
# ============================================================

def obter_pesos_modelo(
    modelo
):
    """
    Como usamos StandardScaler, estes coeficientes
    podem ser comparados em uma escala razoável.

    Positivo:
        aumenta tendência de NÃO sair.

    Negativo:
        aumenta tendência de SAIR.
    """

    logistic = (
        modelo.named_steps[
            "logistic"
        ]
    )

    pesos = (
        logistic.coef_[0]
    )

    resultado = []

    for nome, peso in zip(
        FEATURES_META,
        pesos
    ):

        resultado.append({
            "feature":
                nome,

            "peso":
                float(peso),

            "direcao":
                (
                    "EXCLUSAO"
                    if peso > 0
                    else "PRESENCA"
                ),

            "peso_absoluto":
                abs(
                    float(peso)
                )
        })

    resultado.sort(
        key=lambda item:
            item[
                "peso_absoluto"
            ],
        reverse=True
    )

    return resultado