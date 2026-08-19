# ============================================================
# CONFIGURAÇÃO V5
# ============================================================

# Janelas utilizadas para analisar frequência.
#
# 2 e 3 capturam mudanças muito recentes.
# As janelas maiores dão contexto para saber se essa
# mudança recente é realmente diferente do comportamento
# histórico da dezena.
JANELAS_FREQUENCIA_V5 = [
    2,
    3,
    5,
    6,
    10,
    20,
    50,
    100,
    200,
]


# ============================================================
# TENDÊNCIAS
# ============================================================
#
# Exemplo:
#
# freq_2 = 1.00
# freq_5 = 0.60
#
# tendencia_2_5 = +0.40
#
# Isso indica aceleração recente da presença.
# ============================================================

PARES_TENDENCIA_V5 = [
    # curtíssimo prazo
    (2, 5),
    (2, 6),
    (3, 6),

    # curto x médio
    (3, 10),
    (6, 20),

    # regime recente x histórico
    (6, 50),
    (10, 50),
    (20, 100),

    # longo prazo
    (5, 20),
    (100, 200),
]

# ============================================================
# FEATURES
# ============================================================

USAR_RANKINGS_RELATIVOS = True

USAR_TENDENCIAS = True

USAR_SOBREVIVENCIA_SEQUENCIA = True


# ============================================================
# META MODELO
# ============================================================

LOGISTIC_C = 1.0

LOGISTIC_MAX_ITER = 3000

SEED = 42


# ============================================================
# BACKTEST
# ============================================================

META_TREINO_CONCURSOS = 400

TESTE_FINAL_CONCURSOS = 100

QTD_EXCLUSOES_TESTADAS = [
    4,
    5,
    6,
    7,
]