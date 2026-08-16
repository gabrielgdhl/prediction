import pandas as pd


def carregar_resultados(
    caminho="lotofacil_resultados.xlsx"
):
    """
    Carrega a planilha histórica da Lotofácil
    e retorna somente as 15 colunas das dezenas.
    """

    df = pd.read_excel(caminho)

    colunas_bolas = [
        col
        for col in df.columns
        if "bola" in str(col).lower()
    ]

    if len(colunas_bolas) != 15:
        raise ValueError(
            f"Esperava 15 colunas Bola, "
            f"mas encontrei {len(colunas_bolas)}."
        )

    df_bolas = (
        df[colunas_bolas]
        .dropna()
        .astype(int)
    )

    return df, df_bolas