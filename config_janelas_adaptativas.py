"""Configuração do experimento walk-forward com janelas adaptativas."""

# Casos/features base
JANELA_MINIMA = 200
N_ESTIMATORS = 100
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 10
SEED = 42

# Universo principal. Pontos > 101 só entram se forem explicitamente ativados.
JANELAS_PRINCIPAIS = list(range(2, 102))
JANELAS_EXPERIMENTAIS = [110, 120, 125, 135, 150, 175, 200]
USAR_JANELAS_EXPERIMENTAIS = False

# Seletor causal: força = média exponencial do ganho de log-loss contra 40%.
SELETOR_TOP_K = 12
SELETOR_TEMPERATURA = 0.08
SELETOR_ALPHA = 0.08
SELETOR_PRIOR_FORCA = 0.0
SELETOR_PESO_MINIMO = 1e-6

# Sinal separado: presença em t-1 reduz levemente o score de exclusão.
USAR_SINAL_REPETICAO = True
PESO_REPETICAO_ANTERIOR = 0.025

# Avaliação. Os blocos são apenas rótulos no relatório.
BLOCOS_RELATORIO = 4
TAMANHO_BLOCO_RELATORIO = 100
CENARIOS_EXCLUSOES = [4, 5, 6, 7]
BASELINE_JANELA_FIXA = 94

# Execução/cache
USAR_CACHE_CASOS = True
USAR_CACHE_META = True
ARQUIVO_SAIDA = "resultado_janelas_adaptativas.xlsx"
