import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


class LotofacilDataset:
    """
    Constrói um dataset onde cada linha representa:
    
        concurso + dezena
    
    Exemplo:
    
        concurso | dezena | freq_10 | freq_20 | atraso | nao_saiu
        ---------------------------------------------------------
        1000     | 1      | 5       | 10      | 2      | 0
        1000     | 2      | 3       | 8       | 5      | 1
    """

    def __init__(self, arquivo):
        self.df = pd.read_excel(arquivo)

        self.colunas_bolas = [
            f"Bola{i}"
            for i in range(1, 16)
        ]

        self.resultados = (
            self.df[self.colunas_bolas]
            .astype(int)
            .apply(
                lambda row: set(row),
                axis=1
            )
            .tolist()
        )

        self.total_concursos = len(self.resultados)

    # ---------------------------------------------------------
    # Frequência de uma dezena em uma janela anterior
    # ---------------------------------------------------------

    def frequencia(self, indice, dezena, janela):

        inicio = max(0, indice - janela)

        concursos = self.resultados[inicio:indice]

        if not concursos:
            return 0

        return sum(
            dezena in resultado
            for resultado in concursos
        )

    # ---------------------------------------------------------
    # Atraso atual
    # ---------------------------------------------------------

    def calcular_atraso(self, indice, dezena):

        atraso = 0

        for i in range(indice - 1, -1, -1):

            if dezena in self.resultados[i]:
                return atraso

            atraso += 1

        return atraso

    # ---------------------------------------------------------
    # Intervalos históricos
    # ---------------------------------------------------------

    def intervalos_historicos(self, indice, dezena):

        ocorrencias = []

        for i in range(indice):

            if dezena in self.resultados[i]:
                ocorrencias.append(i)

        if len(ocorrencias) < 2:
            return []

        return [
            ocorrencias[i] - ocorrencias[i - 1]
            for i in range(1, len(ocorrencias))
        ]

    # ---------------------------------------------------------
    # Features de uma dezena antes de um concurso
    # ---------------------------------------------------------

    def features_dezena(self, indice, dezena):

        freq_5 = self.frequencia(
            indice,
            dezena,
            5
        )

        freq_10 = self.frequencia(
            indice,
            dezena,
            10
        )

        freq_20 = self.frequencia(
            indice,
            dezena,
            20
        )

        freq_50 = self.frequencia(
            indice,
            dezena,
            50
        )

        freq_100 = self.frequencia(
            indice,
            dezena,
            100
        )

        freq_200 = self.frequencia(
            indice,
            dezena,
            200
        )

        atraso = self.calcular_atraso(
            indice,
            dezena
        )

        intervalos = self.intervalos_historicos(
            indice,
            dezena
        )

        if intervalos:

            intervalo_medio = np.mean(
                intervalos
            )

            intervalo_mediano = np.median(
                intervalos
            )

            intervalo_desvio = np.std(
                intervalos
            )

            maior_atraso = max(
                intervalos
            )

        else:

            intervalo_medio = 0
            intervalo_mediano = 0
            intervalo_desvio = 0
            maior_atraso = 0

        # Concurso anterior
        apareceu_anterior = (
            indice > 0
            and dezena in self.resultados[indice - 1]
        )

        # Tendências
        tendencia_10_50 = (
            freq_10 / 10
            if freq_50 == 0
            else (freq_10 / 10) / (freq_50 / 50)
        )

        tendencia_20_100 = (
            freq_20 / 20
            if freq_100 == 0
            else (freq_20 / 20) / (freq_100 / 100)
        )

        return {
            "freq_5": freq_5,
            "freq_10": freq_10,
            "freq_20": freq_20,
            "freq_50": freq_50,
            "freq_100": freq_100,
            "freq_200": freq_200,

            "atraso": atraso,

            "intervalo_medio": intervalo_medio,
            "intervalo_mediano": intervalo_mediano,
            "intervalo_desvio": intervalo_desvio,
            "maior_atraso": maior_atraso,

            "apareceu_anterior": int(
                apareceu_anterior
            ),

            "tendencia_10_50": tendencia_10_50,
            "tendencia_20_100": tendencia_20_100,
        }

    # ---------------------------------------------------------
    # Cria dataset completo
    # ---------------------------------------------------------

    def criar_dataset(self, inicio=200):

        registros = []

        for indice in range(
            inicio,
            self.total_concursos
        ):

            resultado_atual = self.resultados[indice]

            for dezena in range(1, 26):

                features = self.features_dezena(
                    indice,
                    dezena
                )

                # Target:
                # 1 = não saiu
                # 0 = saiu
                nao_saiu = int(
                    dezena not in resultado_atual
                )

                registro = {
                    "indice": indice,
                    "concurso": int(
                        self.df.iloc[indice]["Concurso"]
                    ),
                    "dezena": dezena,
                    **features,
                    "nao_saiu": nao_saiu
                }

                registros.append(registro)

        return pd.DataFrame(registros)


# =============================================================
# MODELO
# =============================================================

class ModeloExclusao:

    FEATURES = [
        "freq_5",
        "freq_10",
        "freq_20",
        "freq_50",
        "freq_100",
        "freq_200",

        "atraso",

        "intervalo_medio",
        "intervalo_mediano",
        "intervalo_desvio",
        "maior_atraso",

        "apareceu_anterior",

        "tendencia_10_50",
        "tendencia_20_100",
    ]

    def __init__(self):

        self.modelo = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

    def treinar(self, dataset):

        X = dataset[self.FEATURES]
        y = dataset["nao_saiu"]

        print("Treinando Random Forest...")
        print(f"Observações: {len(dataset):,}")
        print(f"Features: {len(self.FEATURES)}")

        self.modelo.fit(X, y)

        print("Modelo treinado.")

    def prever(self, features):

        """
        Recebe DataFrame contendo uma linha
        para cada dezena.

        Retorna P(não sair).
        """

        X = features[self.FEATURES]

        probabilidades = (
            self.modelo.predict_proba(X)[:, 1]
        )

        resultado = features[
            ["dezena"]
        ].copy()

        resultado["prob_nao_sair"] = (
            probabilidades
        )

        return resultado.sort_values(
            "prob_nao_sair",
            ascending=False
        )

    def importancia_features(self):

        return (
            pd.DataFrame({
                "feature": self.FEATURES,
                "importance":
                    self.modelo.feature_importances_
            })
            .sort_values(
                "importance",
                ascending=False
            )
        )


# =============================================================
# GERADOR DE FEATURES PARA O PRÓXIMO CONCURSO
# =============================================================

def gerar_features_proximo_concurso(
    dataset,
    indice
):

    dados = dataset[
        dataset["indice"] == indice
    ].copy()

    return dados


# =============================================================
# SELECIONAR EXCLUSÕES
# =============================================================

def selecionar_exclusoes(
    modelo,
    features,
    quantidade=7
):

    ranking = modelo.prever(
        features
    )

    exclusoes = (
        ranking
        .head(quantidade)
        ["dezena"]
        .tolist()
    )

    exclusoes = sorted(
        exclusoes
    )

    candidatas = sorted(
        set(range(1, 26))
        - set(exclusoes)
    )

    return (
        exclusoes,
        candidatas,
        ranking
    )


# =============================================================
# EXECUÇÃO
# =============================================================

if __name__ == "__main__":

    arquivo = "Lotofácil.xlsx"

    dataset_builder = LotofacilDataset(
        arquivo
    )

    print(
        f"Concursos encontrados: "
        f"{dataset_builder.total_concursos}"
    )

    dataset = (
        dataset_builder
        .criar_dataset(
            inicio=200
        )
    )

    print()
    print(
        f"Dataset criado: "
        f"{len(dataset):,} linhas"
    )

    # ---------------------------------------------------------
    # IMPORTANTE:
    #
    # Este treinamento ainda é apenas para validar
    # o pipeline.
    #
    # O próximo passo será substituir isso por
    # walk-forward validation.
    # ---------------------------------------------------------

    modelo = ModeloExclusao()

    modelo.treinar(
        dataset
    )

    # Features do último concurso disponível
    ultimo_indice = (
        dataset_builder.total_concursos - 1
    )

    features = gerar_features_proximo_concurso(
        dataset,
        ultimo_indice
    )

    exclusoes, candidatas, ranking = (
        selecionar_exclusoes(
            modelo,
            features,
            quantidade=7
        )
    )

    print()
    print("================================")
    print("RANKING DE EXCLUSÃO")
    print("================================")

    print(
        ranking.to_string(
            index=False
        )
    )

    print()
    print("================================")
    print("7 DEZENAS PARA EXCLUSÃO")
    print("================================")

    print(
        exclusoes
    )

    print()
    print("================================")
    print("18 DEZENAS CANDIDATAS")
    print("================================")

    print(
        candidatas
    )

    print()
    print("================================")
    print("IMPORTÂNCIA DAS FEATURES")
    print("================================")

    print(
        modelo
        .importancia_features()
        .to_string(index=False)
    )