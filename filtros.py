from collections import Counter
from abc import ABC, abstractmethod

import numpy as np


class MotorEstatisticas:

    def __init__(
        self,
        df_historico,
        janela_concursos=15
    ):

        self.df = df_historico

        self.janela = janela_concursos

        self.primos = {
            2, 3, 5, 7, 11,
            13, 17, 19, 23
        }

        self._calcular_frequencias()

        self._calcular_distribuicoes()

    # =====================================================
    # QUENTES / FRIAS
    # =====================================================

    def _calcular_frequencias(self):

        ultimos = (
            self.df
            .tail(self.janela)
            .values
            .flatten()
        )

        contagem = Counter(ultimos)

        self.frequencias = {
            dezena: contagem.get(
                dezena,
                0
            )
            for dezena in range(1, 26)
        }

        ordenadas = sorted(
            self.frequencias,
            key=self.frequencias.get,
            reverse=True
        )

        self.dezenas_quentes = set(
            ordenadas[:5]
        )

        self.dezenas_frias = set(
            ordenadas[-5:]
        )

    # =====================================================
    # DISTRIBUIÇÕES HISTÓRICAS
    # =====================================================

    def _calcular_distribuicoes(self):

        somas = []

        impares = []

        primos = []

        consecutivas = []

        for jogo in self.df.values:

            jogo = sorted(
                map(int, jogo)
            )

            # Soma
            somas.append(
                sum(jogo)
            )

            # Ímpares
            impares.append(
                sum(
                    n % 2 != 0
                    for n in jogo
                )
            )

            # Primos
            primos.append(
                len(
                    set(jogo)
                    & self.primos
                )
            )

            # Consecutivas
            qtd_consecutivas = sum(
                1
                for a, b
                in zip(
                    jogo,
                    jogo[1:]
                )
                if b == a + 1
            )

            consecutivas.append(
                qtd_consecutivas
            )

        self.somas = np.array(
            somas
        )

        self.impares = np.array(
            impares
        )

        self.qtd_primos = np.array(
            primos
        )

        self.consecutivas = np.array(
            consecutivas
        )

        # Percentis
        self.soma_p10 = np.percentile(
            self.somas,
            10
        )

        self.soma_p90 = np.percentile(
            self.somas,
            90
        )

        self.soma_p20 = np.percentile(
            self.somas,
            20
        )

        self.soma_p80 = np.percentile(
            self.somas,
            80
        )