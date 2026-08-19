import sys
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT)
    )


from dados import (
    carregar_resultados
)

from features_v2_reference import (
    GeradorFeaturesV2
)

from features_v5 import (
    calcular_features_v5_concurso
)


# ============================================================
# DADOS
# ============================================================

caminho_excel = (
    ROOT
    / "lotofacil_resultados.xlsx"
)

df, df_bolas = (
    carregar_resultados(
        caminho_excel
    )
)

gerador = (
    GeradorFeaturesV2(
        df_bolas
    )
)


# ============================================================
# ÚLTIMO ESTADO CONHECIDO
# ============================================================

indice_estado = (
    gerador.total_sorteios
    - 1
)


extras = (
    calcular_features_v5_concurso(
        matriz_binaria=
            gerador.matriz_binaria,

        indice_estado=
            indice_estado
    )
)


# ============================================================
# MOSTRAR
# ============================================================

for dezena in range(
    1,
    26
):

    item = extras[
        dezena
    ]

    print()
    print(
        f"DEZENA {dezena:02d}"
    )

    print(
        f"freq_2: "
        f"{item['freq_2']:.3f}"
    )

    print(
        f"freq_3: "
        f"{item['freq_3']:.3f}"
    )

    print(
        f"freq_5: "
        f"{item['freq_5']:.3f}"
    )

    print(
        f"tendencia_2_5: "
        f"{item['tendencia_2_5']:+.3f}"
    )

    print(
        f"sequencia: "
        f"{item['sequencia_atual_v5']}"
    )

    print(
        f"P sobreviver: "
        f"{item['prob_sobreviver_sequencia']:.3f}"
    )

    print(
        f"lift sobrevivência: "
        f"{item['lift_sobrevivencia']:+.3f}"
    )

    print(
        f"amostras: "
        f"{item['amostras_sobrevivencia']}"
    )

    print(
        f"rank sobrevivência: "
        f"{item['rank_sobrevivencia']:.3f}"
    )