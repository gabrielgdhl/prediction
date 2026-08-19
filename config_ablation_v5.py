# ============================================================
# CONFIGURAÇÃO - ABLATION V5
# ============================================================


# ============================================================
# WALK-FORWARD
# ============================================================

JANELA_MINIMA = 200

META_TREINO_CONCURSOS = 100


# ------------------------------------------------------------
# Quantos blocos temporais queremos avaliar.
#
# 1:
#   apenas os últimos 100
#
# 4:
#   últimos 400 divididos em 4 blocos independentes
#
# Minha recomendação:
#
# primeiro:
#   BLOCOS_TESTE = 1
#
# depois da validação:
#   BLOCOS_TESTE = 4
# ------------------------------------------------------------

BLOCOS_TESTE = 4

TAMANHO_BLOCO_TESTE = 100


# ============================================================
# V2
# ============================================================

N_ESTIMATORS = 100

MAX_DEPTH = 8

MIN_SAMPLES_LEAF = 10

SEED = 42


# ============================================================
# CENÁRIOS
# ============================================================

CENARIOS_EXCLUSOES = [
    4,
    5,
    6,
    7,
]


# ============================================================
# META-MODELO
# ============================================================

LOGISTIC_C = 1.0

LOGISTIC_MAX_ITER = 3000


# ============================================================
# GRUPOS DE FEATURES
#
# IMPORTANTE:
#
# O V4 já será usado como baseline.
#
# Aqui colocamos somente as features EXTRAS.
# ============================================================


GRUPO_RANKING_V2 = [
    "rank_v2",
    "gap_media_v2",
    "gap_top1_v2",
    "zscore_v2",
]


GRUPO_TENDENCIA_CURTA = [
    "rank_tendencia_2_5",
    "rank_tendencia_3_10",
]


GRUPO_TENDENCIA_COMPLETA = [
    "rank_tendencia_2_5",
    "rank_tendencia_3_10",
    "rank_tendencia_5_20",
    "rank_tendencia_10_50",
]


GRUPO_SEQUENCIA = [
    "prob_sobreviver_sequencia",
    "lift_sobrevivencia",
    "log_amostras_sobrevivencia",
    "prob_terminar_agora",
]


GRUPO_FREQUENCIA_CURTA = [
    "freq_2",
    "freq_3",

    "rank_freq_2",
    "rank_freq_3",
]


GRUPO_PRESENCA_RELATIVA = [
    "rank_presenca",
    "gap_media_presenca",
]


GRUPO_EXAUSTAO_RELATIVA = [
    "rank_exaustao",
]

GRUPO_FREQUENCIA_MICRO = [
    "freq_2",
    "freq_3",
    "freq_6",

    "rank_freq_2",
    "rank_freq_3",
    "rank_freq_6",
]


GRUPO_FREQUENCIA_MESO = [
    "freq_10",
    "freq_20",

    "rank_freq_10",
    "rank_freq_20",
]


GRUPO_BASELINE_LONGO = [
    "freq_50",
    "freq_100",
    "freq_200",

    "rank_freq_50",
    "rank_freq_100",
    "rank_freq_200",
]


GRUPO_TENDENCIA_MICRO = [
    "rank_tendencia_2_5",
    "rank_tendencia_2_6",
    "rank_tendencia_3_6",
]


GRUPO_MUDANCA_REGIME = [
    "rank_tendencia_3_10",
    "rank_tendencia_6_20",
    "rank_tendencia_6_50",
    "rank_tendencia_10_50",
    "rank_tendencia_20_100",
]


GRUPO_TENDENCIA_LONGA = [
    "rank_tendencia_5_20",
    "rank_tendencia_100_200",
]


# ============================================================
# EXPERIMENTOS
# ============================================================

EXPERIMENTOS = {

    "V4": [],

    "V4_RANK_TEND_FREQ_ANTERIOR":
        (
            GRUPO_RANKING_V2
            + [
                "rank_tendencia_2_5",
                "rank_tendencia_3_10",
            ]
            + [
                "freq_2",
                "freq_3",
                "rank_freq_2",
                "rank_freq_3",
            ]
        ),

    "V4_MICRO_LONGO":
        (
            GRUPO_RANKING_V2
            + GRUPO_FREQUENCIA_MICRO
            + GRUPO_TENDENCIA_MICRO
            + GRUPO_BASELINE_LONGO
        ),

    "V4_MULTIESCALA":
        (
            GRUPO_RANKING_V2
            + GRUPO_FREQUENCIA_MICRO
            + GRUPO_FREQUENCIA_MESO
            + GRUPO_BASELINE_LONGO
            + GRUPO_TENDENCIA_MICRO
            + GRUPO_MUDANCA_REGIME
            + GRUPO_TENDENCIA_LONGA
        ),
}