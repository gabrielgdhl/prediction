import numpy as np

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    StandardScaler,
)


from ranking_v3 import (
    calcular_score_presenca,
    analisar_exaustao,
)

from ranking_v4 import (
    FEATURES_META,
    construir_features_meta,
)

from config_v5 import (
    JANELAS_FREQUENCIA_V5,
    PARES_TENDENCIA_V5,
    LOGISTIC_C,
    LOGISTIC_MAX_ITER,
    SEED,
)


# ============================================================
# FEATURES RELATIVAS
# ============================================================

FEATURES_RELATIVAS_V5 = [
    "rank_v2",
    "gap_media_v2",
    "gap_top1_v2",
    "zscore_v2",

    "rank_presenca",
    "gap_media_presenca",

    "rank_exaustao",
]


# ============================================================
# JANELAS
# ============================================================

FEATURES_JANELAS_V5 = []

for janela in JANELAS_FREQUENCIA_V5:

    FEATURES_JANELAS_V5.extend([
        f"freq_{janela}",
        f"rank_freq_{janela}",
    ])


# ============================================================
# TENDÊNCIAS
# ============================================================

FEATURES_TENDENCIA_V5 = []

for curta, longa in PARES_TENDENCIA_V5:

    FEATURES_TENDENCIA_V5.extend([
        f"tendencia_{curta}_{longa}",
        f"rank_tendencia_{curta}_{longa}",
    ])


# ============================================================
# SEQUÊNCIA
# ============================================================

FEATURES_SEQUENCIA_V5 = [
    "prob_sobreviver_sequencia",
    "lift_sobrevivencia",
    "log_amostras_sobrevivencia",
    "prob_terminar_agora",
    "rank_sobrevivencia",
]


# ============================================================
# TODAS AS FEATURES EXTRAS
# ============================================================

FEATURES_EXTRAS_CANDIDATAS = (
    FEATURES_RELATIVAS_V5
    + FEATURES_JANELAS_V5
    + FEATURES_TENDENCIA_V5
    + FEATURES_SEQUENCIA_V5
)


# ============================================================
# REMOVER DUPLICADAS DO V4
#
# Exemplo:
#
# V4 já possui freq_5.
#
# Portanto V5 NÃO deve inserir freq_5 novamente.
# ============================================================

FEATURES_EXTRAS_V5 = []

for feature in FEATURES_EXTRAS_CANDIDATAS:

    if feature in FEATURES_META:
        continue

    if feature in FEATURES_EXTRAS_V5:
        continue

    FEATURES_EXTRAS_V5.append(
        feature
    )


FEATURES_META_V5 = (
    list(FEATURES_META)
    + FEATURES_EXTRAS_V5
)


# ============================================================
# VALIDAÇÃO
# ============================================================

def validar_features_unicas(
    features
):

    duplicadas = sorted({
        feature
        for feature in features
        if features.count(feature) > 1
    })

    if duplicadas:

        raise ValueError(
            "Features duplicadas detectadas: "
            + ", ".join(
                duplicadas
            )
        )


validar_features_unicas(
    FEATURES_META_V5
)


# ============================================================
# RANKING
# ============================================================

def calcular_rank_desc(
    valores
):

    valores = np.asarray(
        valores,
        dtype=float
    )

    ordem = np.argsort(
        -valores
    )

    ranks = np.empty(
        len(valores),
        dtype=np.int16
    )

    ranks[ordem] = (
        np.arange(
            len(valores)
        )
        + 1
    )

    return ranks


def normalizar_rank(
    rank,
    total=25
):

    if total <= 1:
        return 0.0

    return (
        (rank - 1)
        / (total - 1)
    )


# ============================================================
# CONTEXTO RELATIVO
# ============================================================

def construir_contexto_relativo(
    ranking_v2,
    features_por_dezena
):

    ranking_map = {
        item["dezena"]:
            item

        for item in ranking_v2
    }

    # ========================================================
    # V2
    # ========================================================

    probs_v2 = np.asarray(
        [
            ranking_map[
                dezena
            ][
                "prob_nao_sair"
            ]

            for dezena
            in range(
                1,
                26
            )
        ],
        dtype=float
    )

    media_v2 = float(
        np.mean(
            probs_v2
        )
    )

    desvio_v2 = float(
        np.std(
            probs_v2
        )
    )

    top1_v2 = float(
        np.max(
            probs_v2
        )
    )

    ranks_v2 = (
        calcular_rank_desc(
            probs_v2
        )
    )

    # ========================================================
    # PRESENÇA
    # ========================================================

    scores_presenca = []

    scores_exaustao = []

    for dezena in range(
        1,
        26
    ):

        features = (
            features_por_dezena[
                dezena
            ]
        )

        presenca = (
            calcular_score_presenca(
                features
            )
        )

        exaustao = (
            analisar_exaustao(
                features,
                lift_minimo=-1.0,
                amostras_minimas=0
            )
        )

        scores_presenca.append(
            presenca[
                "score_presenca"
            ]
        )

        scores_exaustao.append(
            exaustao[
                "lift_exaustao"
            ]
        )

    scores_presenca = np.asarray(
        scores_presenca,
        dtype=float
    )

    scores_exaustao = np.asarray(
        scores_exaustao,
        dtype=float
    )

    media_presenca = float(
        np.mean(
            scores_presenca
        )
    )

    ranks_presenca = (
        calcular_rank_desc(
            scores_presenca
        )
    )

    ranks_exaustao = (
        calcular_rank_desc(
            scores_exaustao
        )
    )

    # ========================================================
    # RESULTADO
    # ========================================================

    resultado = {}

    for indice, dezena in enumerate(
        range(
            1,
            26
        )
    ):

        resultado[
            dezena
        ] = {

            "rank_v2":
                normalizar_rank(
                    int(
                        ranks_v2[
                            indice
                        ]
                    )
                ),

            "gap_media_v2":
                float(
                    probs_v2[
                        indice
                    ]
                    - media_v2
                ),

            "gap_top1_v2":
                float(
                    probs_v2[
                        indice
                    ]
                    - top1_v2
                ),

            "zscore_v2":
                float(
                    (
                        probs_v2[
                            indice
                        ]
                        - media_v2
                    )
                    / (
                        desvio_v2
                        + 1e-8
                    )
                ),

            "rank_presenca":
                normalizar_rank(
                    int(
                        ranks_presenca[
                            indice
                        ]
                    )
                ),

            "gap_media_presenca":
                float(
                    scores_presenca[
                        indice
                    ]
                    - media_presenca
                ),

            "rank_exaustao":
                normalizar_rank(
                    int(
                        ranks_exaustao[
                            indice
                        ]
                    )
                ),
        }

    return resultado


# ============================================================
# TRANSFORMAR FEATURES EXTRAS EM DICT ÚNICO
# ============================================================

def construir_extras_completos(
    ranking_v2,
    features_por_dezena,
    extras_por_dezena
):

    contexto = (
        construir_contexto_relativo(
            ranking_v2=
                ranking_v2,

            features_por_dezena=
                features_por_dezena
        )
    )

    resultado = {}

    for dezena in range(
        1,
        26
    ):

        dados = {}

        dados.update(
            extras_por_dezena[
                dezena
            ]
        )

        dados.update(
            contexto[
                dezena
            ]
        )

        resultado[
            dezena
        ] = dados

    return resultado


# ============================================================
# UMA LINHA
# ============================================================

def construir_features_meta_custom(
    ranking_v2_item,
    features_dezena,
    extras_dezena,
    features_extras
):

    base_v4 = (
        construir_features_meta(
            ranking_v2_item=
                ranking_v2_item,

            features_dezena=
                features_dezena
        )
    )

    extras = []

    for nome in features_extras:

        if nome in FEATURES_META:

            # Já existe dentro do V4.
            continue

        if nome not in extras_dezena:

            raise KeyError(
                f"Feature extra "
                f"'{nome}' não encontrada."
            )

        extras.append(
            float(
                extras_dezena[
                    nome
                ]
            )
        )

    return np.concatenate([
        np.asarray(
            base_v4,
            dtype=np.float64
        ),

        np.asarray(
            extras,
            dtype=np.float64
        ),
    ])


# ============================================================
# FEATURES FINAIS DO EXPERIMENTO
# ============================================================

def obter_features_modelo(
    features_extras
):

    extras_filtradas = []

    for feature in features_extras:

        if feature in FEATURES_META:
            continue

        if feature in extras_filtradas:
            continue

        extras_filtradas.append(
            feature
        )

    resultado = (
        list(FEATURES_META)
        + extras_filtradas
    )

    validar_features_unicas(
        resultado
    )

    return resultado


# ============================================================
# MATRIZ CUSTOM
# ============================================================

def construir_matriz_custom(
    ranking_v2,
    features_por_dezena,
    extras_por_dezena,
    features_extras
):

    ranking_map = {
        item["dezena"]:
            item

        for item
        in ranking_v2
    }

    extras_completos = (
        construir_extras_completos(
            ranking_v2=
                ranking_v2,

            features_por_dezena=
                features_por_dezena,

            extras_por_dezena=
                extras_por_dezena
        )
    )

    linhas = []

    for dezena in range(
        1,
        26
    ):

        linha = (
            construir_features_meta_custom(
                ranking_v2_item=
                    ranking_map[
                        dezena
                    ],

                features_dezena=
                    features_por_dezena[
                        dezena
                    ],

                extras_dezena=
                    extras_completos[
                        dezena
                    ],

                features_extras=
                    features_extras
            )
        )

        linhas.append(
            linha
        )

    return np.asarray(
        linhas,
        dtype=np.float64
    )


# ============================================================
# V5 NORMAL
# ============================================================

def construir_matriz_v5(
    ranking_v2,
    features_por_dezena,
    extras_por_dezena
):

    return (
        construir_matriz_custom(
            ranking_v2=
                ranking_v2,

            features_por_dezena=
                features_por_dezena,

            extras_por_dezena=
                extras_por_dezena,

            features_extras=
                FEATURES_EXTRAS_V5
        )
    )


# ============================================================
# MODELO
# ============================================================

def criar_modelo_meta(
    c=LOGISTIC_C,
    max_iter=LOGISTIC_MAX_ITER
):

    return Pipeline([
        (
            "scaler",

            StandardScaler()
        ),

        (
            "logistic",

            LogisticRegression(
                C=c,
                max_iter=max_iter,
                random_state=SEED
            )
        )
    ])


def treinar_modelo_meta(
    X,
    y
):

    modelo = (
        criar_modelo_meta()
    )

    modelo.fit(
        X,
        y
    )

    return modelo


def treinar_v5(
    X,
    y
):

    return (
        treinar_modelo_meta(
            X,
            y
        )
    )


# ============================================================
# RANKING CUSTOM
# ============================================================

def criar_ranking_custom(
    modelo,
    ranking_v2,
    features_por_dezena,
    extras_por_dezena,
    features_extras
):

    X = (
        construir_matriz_custom(
            ranking_v2=
                ranking_v2,

            features_por_dezena=
                features_por_dezena,

            extras_por_dezena=
                extras_por_dezena,

            features_extras=
                features_extras
        )
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

    indice_nao_sair = int(
        np.where(
            classes == 1
        )[0][0]
    )

    probs = (
        probabilidades[
            :,
            indice_nao_sair
        ]
    )

    ranking = []

    for dezena, prob in zip(
        range(
            1,
            26
        ),
        probs
    ):

        ranking.append({
            "dezena":
                dezena,

            "prob_nao_sair":
                float(
                    prob
                ),

            "prob_sair":
                float(
                    1.0 - prob
                )
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
# RANKING V5
# ============================================================

def criar_ranking_v5(
    modelo,
    ranking_v2,
    features_por_dezena,
    extras_por_dezena
):

    return (
        criar_ranking_custom(
            modelo=
                modelo,

            ranking_v2=
                ranking_v2,

            features_por_dezena=
                features_por_dezena,

            extras_por_dezena=
                extras_por_dezena,

            features_extras=
                FEATURES_EXTRAS_V5
        )
    )


# ============================================================
# PESOS
# ============================================================

def obter_pesos_custom(
    modelo,
    features_extras
):

    nomes = (
        obter_features_modelo(
            features_extras
        )
    )

    pesos = (
        modelo.named_steps[
            "logistic"
        ].coef_[0]
    )

    if len(
        nomes
    ) != len(
        pesos
    ):

        raise ValueError(
            "Quantidade de nomes de features "
            "não corresponde aos coeficientes."
        )

    resultado = []

    for nome, peso in zip(
        nomes,
        pesos
    ):

        peso = float(
            peso
        )

        resultado.append({
            "feature":
                nome,

            "peso":
                peso,

            "peso_absoluto":
                abs(
                    peso
                ),

            "direcao":
                (
                    "EXCLUSAO"
                    if peso > 0
                    else "PRESENCA"
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


def obter_pesos_v5(
    modelo
):

    return (
        obter_pesos_custom(
            modelo,
            FEATURES_EXTRAS_V5
        )
    )