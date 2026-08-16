import numpy as np
import pandas as pd
from itertools import combinations
import random
import time

class LotofacilBacktester:
    def __init__(self, df_resultados, n_exclusoes=7, jogos_alvo=[20, 30, 50, 100]):
        """
        df_resultados: DataFrame com as colunas Bola1 a Bola15 (ordenadas temporalmente)
        """
        self.df = df_resultados
        self.total_concursos = len(df_resultados)
        self.n_exclusoes = n_exclusoes
        self.n_candidatas = 25 - n_exclusoes
        self.jogos_alvo = jogos_alvo
        
        # Converte para matriz binária de forma antecipada para performance
        # Linhas: concursos, Colunas: dezenas (0 a 24 representando 1 a 25)
        self.matriz_sorteios = np.zeros((self.total_concursos, 25), dtype=int)
        for i, sorteio in enumerate(self.df.values):
            for bola in sorteio:
                self.matriz_sorteios[i, bola - 1] = 1
                
        # Gera todas as combinações de 15 possíveis a partir de 18 (816 combinações)
        # Convertidas para bitmasks para cálculo rápido de distância (XOR)
        self.combinacoes_base = [
            sum(1 << (n - 1) for n in comb) 
            for comb in combinations(range(1, self.n_candidatas + 1), 15)
        ]

    def _calcular_features_passado(self, limite_idx):
        """
        Calcula estatísticas usando APENAS dados de 0 até limite_idx - 1 (Zero Data Leakage).
        Retorna um dicionário com o "Score de Exclusão" (maior = mais provável não sair)
        """
        matriz_passado = self.matriz_sorteios[:limite_idx]
        
        # 1. Frequência (últimos 15 concursos)
        janela_freq = max(0, limite_idx - 15)
        frequencia = np.sum(matriz_passado[janela_freq:], axis=0)
        
        # 2. Atraso atual
        atraso = np.zeros(25, dtype=int)
        for dezena in range(25):
            atr = 0
            for i in range(limite_idx - 1, -1, -1):
                if matriz_passado[i, dezena] == 1:
                    break
                atr += 1
            atraso[dezena] = atr
            
        # Heurística Simples para exclusão:
        # Aposta que dezenas que saíram MUITO recentemente e têm BAIXO atraso vão "descansar"
        # (Isso é o que será testado. Pode ser invertido depois).
        scores_exclusao = frequencia - (atraso * 0.5) 
        
        ranking = [(dezena + 1, scores_exclusao[dezena]) for dezena in range(25)]
        # Ordena do maior score (piores dezenas) para o menor
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking

    def _fechamento_guloso_bitmask(self, dezenas_candidatas, qtd_jogos):
        """
        Greedy max-min: Maximiza a diversidade (distância de Hamming) entre os jogos.
        Mapeia a combinação 1-18 para as dezenas reais antes de retornar.
        """
        if qtd_jogos >= len(self.combinacoes_base):
            selecionados_bits = self.combinacoes_base
        else:
            selecionados_bits = [self.combinacoes_base[0]]
            candidatos_restantes = set(self.combinacoes_base[1:])
            
            while len(selecionados_bits) < qtd_jogos:
                melhor_candidato = None
                max_distancia_minima = -1
                
                for cand in candidatos_restantes:
                    # Distância = quantidade de bits diferentes (XOR -> bit count)
                    distancia_minima = min((cand ^ sel).bit_count() for sel in selecionados_bits)
                    
                    if distancia_minima > max_distancia_minima:
                        max_distancia_minima = distancia_minima
                        melhor_candidato = cand
                
                selecionados_bits.append(melhor_candidato)
                candidatos_restantes.remove(melhor_candidato)
        
        # Traduz a bitmask (1 a 18) para as dezenas reais (candidatas selecionadas)
        jogos_reais = []
        for mask in selecionados_bits:
            jogo = []
            for i in range(self.n_candidatas):
                if (mask & (1 << i)):
                    jogo.append(dezenas_candidatas[i])
            jogos_reais.append(set(jogo))
            
        return jogos_reais

    def executar_walk_forward(self, inicio_idx=100, qtd_concursos_teste=50):
        """
        Percorre o tempo, prevê, filtra e confere.
        """
        resultados_backtest = []
        todas_dezenas = set(range(1, 26))
        
        fim_idx = min(inicio_idx + qtd_concursos_teste, self.total_concursos)
        
        print(f"Iniciando Walk-Forward Backtest (Concursos {inicio_idx} a {fim_idx - 1})...\n")
        start_time = time.time()
        
        for idx in range(inicio_idx, fim_idx):
            # 1. O que realmente aconteceu no concurso atual
            sorteio_real = set(np.where(self.matriz_sorteios[idx] == 1)[0] + 1)
            nao_sorteadas_reais = todas_dezenas - sorteio_real
            
            # ==============================================================
            # MODELO A: ESTATÍSTICA SIMPLES
            # ==============================================================
            ranking = self._calcular_features_passado(idx)
            exclusões_stats = set(d for d, score in ranking[:self.n_exclusoes])
            acertos_exclusao_stats = len(exclusões_stats.intersection(nao_sorteadas_reais))
            
            # ==============================================================
            # MODELO B: BASELINE ALEATÓRIO
            # ==============================================================
            exclusoes_aleatorias = set(random.sample(list(todas_dezenas), self.n_exclusoes))
            acertos_exclusao_rand = len(exclusoes_aleatorias.intersection(nao_sorteadas_reais))
            
            # 2. Separa as 18 candidatas (usando o modelo estatístico para o fechamento)
            candidatas_stats = sorted(list(todas_dezenas - exclusões_stats))
            
            # 3. Executa fechamento e avaliação de prêmios
            linha_resultado = {
                'concurso': idx + 1,
                'acertos_exclusao_stats': acertos_exclusao_stats,
                'acertos_exclusao_rand': acertos_exclusao_rand
            }
            
            for qtd in self.jogos_alvo:
                jogos = self._fechamento_guloso_bitmask(candidatas_stats, qtd)
                
                # Confere os acertos
                acertos_jogos = [len(jogo.intersection(sorteio_real)) for jogo in jogos]
                
                linha_resultado[f'melhor_jogo_{qtd}'] = max(acertos_jogos)
                linha_resultado[f'{qtd}j_11+'] = sum(1 for a in acertos_jogos if a >= 11)
                linha_resultado[f'{qtd}j_12+'] = sum(1 for a in acertos_jogos if a >= 12)
                linha_resultado[f'{qtd}j_13+'] = sum(1 for a in acertos_jogos if a >= 13)
                linha_resultado[f'{qtd}j_14+'] = sum(1 for a in acertos_jogos if a >= 14)
                linha_resultado[f'{qtd}j_15'] = sum(1 for a in acertos_jogos if a == 15)
                
            resultados_backtest.append(linha_resultado)
            
            if (idx - inicio_idx + 1) % 10 == 0:
                print(f"Processado: {idx - inicio_idx + 1}/{qtd_concursos_teste} concursos...")
                
        df_resultados = pd.DataFrame(resultados_backtest)
        
        print(f"\nBacktest concluído em {time.time() - start_time:.2f} segundos.")
        return df_resultados