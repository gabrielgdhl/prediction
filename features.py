import numpy as np
import pandas as pd

class GeradorEstatisticasAvancadas:
    def __init__(self, df_bolas):
        self.df_bolas = df_bolas
        self.total_sorteios = len(df_bolas)
        self.matriz_binaria = self._criar_matriz_binaria()

    def _criar_matriz_binaria(self):
        matriz = np.zeros((self.total_sorteios, 25), dtype=int)
        for i, sorteio in enumerate(self.df_bolas.values):
            for bola in sorteio:
                matriz[i, bola - 1] = 1
        return matriz

    def calcular_features_no_indice(self, indice):
        """
        Calcula o vetor de features para as 25 dezenas no dado 'indice' do concurso,
        utilizando estritamente apenas dados anteriores (sem data leakage).
        """
        matriz_passado = self.matriz_binaria[:indice + 1]
        tamanho_passado = len(matriz_passado)
        
        features_totais = []
        
        # Estado do Concurso Anterior (para capturar repetição e global)
        sorteio_anterior = self.matriz_binaria[indice] if indice >= 0 else np.zeros(25)
        
        for dezena in range(25):
            # Histórico específico da dezena no passado (corrigido o recuo aqui)
            serie_dezena = matriz_passado[:, dezena]
            
            # --- Frequências em Janelas Múltiplas ---
            freq_5 = np.sum(serie_dezena[-5:]) if tamanho_passado >= 5 else np.sum(serie_dezena)
            freq_10 = np.sum(serie_dezena[-10:]) if tamanho_passado >= 10 else np.sum(serie_dezena)
            freq_20 = np.sum(serie_dezena[-20:]) if tamanho_passado >= 20 else np.sum(serie_dezena)
            freq_50 = np.sum(serie_dezena[-50:]) if tamanho_passado >= 50 else np.sum(serie_dezena)
            freq_100 = np.sum(serie_dezena[-100:]) if tamanho_passado >= 100 else np.sum(serie_dezena)
            freq_200 = np.sum(serie_dezena[-200:]) if tamanho_passado >= 200 else np.sum(serie_dezena)
            freq_historica = np.sum(serie_dezena)
            
            # --- Atrasos e Estatísticas de Intervalo ---
            atraso_atual = 0
            for i in range(tamanho_passado - 1, -1, -1):
                if serie_dezena[i] == 1:
                    break
                atraso_atual += 1
                
            # Calcula todos os intervalos (atrasos passados) para média, mediana e desvio
            intervalos = []
            ultimo_idx = -1
            for i, val in enumerate(serie_dezena):
                if val == 1:
                    if ultimo_idx != -1:
                        intervalos.append(i - ultimo_idx - 1)
                    ultimo_idx = i
                    
            if len(intervalos) > 0:
                media_intervalo = np.mean(intervalos)
                mediana_intervalo = np.median(intervalos)
                desvio_intervalo = np.std(intervalos)
                max_atraso_historico = max(intervalos + [atraso_atual])
            else:
                media_intervalo = 0.0
                mediana_intervalo = 0.0
                desvio_intervalo = 0.0
                max_atraso_historico = atraso_atual

            # --- Comportamento de Repetição ---
            apareceu_ultimo = int(sorteio_anterior[dezena] == 1)
            
            # --- Z-Score do Atraso (Normalização de Ruptura) ---
            z_score_atraso = (atraso_atual - media_intervalo) / (desvio_intervalo + 1e-5)
            
            # Agrupa todas as features desta dezena
            features_dezena = [
                freq_5,
                freq_10,
                freq_20,
                freq_50,
                freq_100,
                freq_200,
                freq_historica,
                atraso_atual,
                max_atraso_historico,
                media_intervalo,
                mediana_intervalo,
                desvio_intervalo,
                z_score_atraso,
                apareceu_ultimo
            ]
            
            features_totais.extend(features_dezena)
            
        return np.array(features_totais)

    def construir_dataset_ml(self, janela_minima=200):
        """
        Constrói o dataset (X, y) para treinamento do Machine Learning.
        y = 1 se a dezena SAIU no concurso seguinte, 0 se NÃO SAIU.
        """
        X = []
        y = []
        
        for i in range(janela_minima, self.total_sorteios - 1):
            features_atuais = self.calcular_features_no_indice(i)
            target_futuro = self.matriz_binaria[i + 1] # 1 se saiu, 0 se não saiu
            
            X.append(features_atuais)
            y.append(target_futuro)
            
        return np.array(X), np.array(y)