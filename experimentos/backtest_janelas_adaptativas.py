"""Backtest causal de janelas meta adaptativas (2..101 por padrão).

Invariantes:
* o caso do alvo t usa features calculadas no estado t-1;
* cada meta-modelo usa apenas casos [t-w, t);
* a força de uma janela em t contém desempenho somente até t-1;
* blocos são criados após as previsões e servem apenas ao relatório;
* caches guardam casos/matrizes, nunca modelos ou previsões futuras.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cache_dataset import obter_dataset_v2
from cache_experimento import CacheExperimento, assinatura
from config_ablation_v5 import (
    GRUPO_BASELINE_LONGO, GRUPO_FREQUENCIA_MESO, GRUPO_FREQUENCIA_MICRO,
    GRUPO_MUDANCA_REGIME, GRUPO_RANKING_V2, GRUPO_TENDENCIA_LONGA,
    GRUPO_TENDENCIA_MICRO,
)
from config_janelas_adaptativas import *
from dados import carregar_resultados
from features_v2_reference import GeradorFeaturesV2
from features_v5 import calcular_features_v5_concurso
from ranking_v5 import construir_matriz_custom, treinar_modelo_meta


FEATURES_MULTIESCALA = (
    GRUPO_RANKING_V2 + GRUPO_FREQUENCIA_MICRO + GRUPO_FREQUENCIA_MESO
    + GRUPO_BASELINE_LONGO + GRUPO_TENDENCIA_MICRO
    + GRUPO_MUDANCA_REGIME + GRUPO_TENDENCIA_LONGA
)


def janelas_candidatas() -> list[int]:
    janelas = list(JANELAS_PRINCIPAIS)
    if USAR_JANELAS_EXPERIMENTAIS:
        janelas += list(JANELAS_EXPERIMENTAIS)
    janelas = sorted(set(int(x) for x in janelas))
    if not janelas or min(janelas) < 2 or max(janelas) > 200:
        raise ValueError("Janelas devem estar no intervalo 2..200.")
    return janelas


def criar_modelo_v2():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=SEED,
        n_jobs=-1, class_weight="balanced_subsample",
    )


def criar_ranking_v2(modelo, X_teste, dezenas_teste):
    indice_1 = int(np.where(modelo.classes_ == 1)[0][0])
    probs_sair = modelo.predict_proba(X_teste)[:, indice_1]
    return sorted(
        ({"dezena": int(d), "prob_sair": float(p),
          "prob_nao_sair": float(1.0 - p)}
         for d, p in zip(dezenas_teste, probs_sair)),
        key=lambda x: x["prob_nao_sair"], reverse=True,
    )


def features_por_dezena(X_teste, dezenas_teste):
    nomes = GeradorFeaturesV2.nomes_features()
    return {
        int(dezena): {nome: float(valor) for nome, valor in zip(nomes, linha)}
        for linha, dezena in zip(X_teste, dezenas_teste)
    }


def preparar_dataset():
    excel = ROOT / "lotofacil_resultados.xlsx"
    features = ROOT / "features_v2_reference.py"
    df, df_bolas = carregar_resultados(excel)
    gerador, X, y, indices, dezenas, binaria = obter_dataset_v2(
        excel, features, df_bolas, GeradorFeaturesV2, JANELA_MINIMA
    )
    if gerador is None:
        gerador = type("GeradorCache", (), {})()
        gerador.total_sorteios = len(binaria)
        gerador.matriz_binaria = binaria
    return df, gerador, X, y, indices, dezenas


def gerar_caso(indice, df, gerador, X, y, indices_target, dezenas):
    treino = indices_target < indice
    teste = indices_target == indice
    if int(teste.sum()) != 25 or not treino.any():
        raise ValueError(f"Caso {indice} sem treino ou sem as 25 dezenas.")
    modelo = criar_modelo_v2()
    modelo.fit(X[treino], y[treino])
    X_teste, dezenas_teste = X[teste], dezenas[teste]
    sorteadas = set(np.flatnonzero(gerador.matriz_binaria[indice]) + 1)
    concurso = int(df.iloc[indice]["Concurso"]) if "Concurso" in df else indice + 1
    return {
        "indice": indice, "concurso": concurso,
        "ranking_v2": criar_ranking_v2(modelo, X_teste, dezenas_teste),
        "features_por_dezena": features_por_dezena(X_teste, dezenas_teste),
        "extras_v5": calcular_features_v5_concurso(
            gerador.matriz_binaria, indice_estado=indice - 1
        ),
        "sorteadas": sorteadas,
        "nao_sorteadas": set(range(1, 26)) - sorteadas,
        "sorteadas_anterior": set(np.flatnonzero(gerador.matriz_binaria[indice - 1]) + 1),
    }


def criar_cache() -> CacheExperimento:
    deps = {
        "planilha": ROOT / "lotofacil_resultados.xlsx",
        "features_v2": ROOT / "features_v2_reference.py",
        "features_v5": ROOT / "features_v5.py",
        "ranking_v5": ROOT / "ranking_v5.py",
        "config_features": ROOT / "config_ablation_v5.py",
        "config_experimento": ROOT / "config_janelas_adaptativas.py",
        "script": Path(__file__),
    }
    params = {
        "janela_minima": JANELA_MINIMA, "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH, "min_samples_leaf": MIN_SAMPLES_LEAF,
        "seed": SEED, "features_meta": FEATURES_MULTIESCALA,
    }
    return CacheExperimento(ROOT, "janelas_adaptativas", assinatura(deps, params))


def preparar_casos(cache: CacheExperimento):
    df, gerador, X, y, indices_target, dezenas = preparar_dataset()
    targets = sorted(int(x) for x in np.unique(indices_target))
    primeiro = targets[1]
    ultimo = gerador.total_sorteios - 1
    casos = cache.carregar_casos() if USAR_CACHE_CASOS else None
    casos = casos or {}
    faltantes = [i for i in range(primeiro, ultimo + 1) if i not in casos]
    print(f"Casos: {len(casos)} em cache; {len(faltantes)} a calcular.")
    for numero, indice in enumerate(faltantes, 1):
        casos[indice] = gerar_caso(
            indice, df, gerador, X, y, indices_target, dezenas
        )
        if numero % 25 == 0 or numero == len(faltantes):
            print(f"  casos {numero}/{len(faltantes)}")
            if USAR_CACHE_CASOS:
                cache.salvar_casos(casos)
    return casos, gerador.total_sorteios


def preparar_matrizes_meta(casos: dict, cache: CacheExperimento):
    carregado = cache.carregar_meta() if USAR_CACHE_META else None
    if carregado is not None:
        indices, X_meta, y_meta = carregado
        if list(indices) == sorted(casos):
            print(f"Matrizes meta carregadas: {X_meta.shape}.")
            return indices, X_meta, y_meta
    indices = np.asarray(sorted(casos), dtype=np.int32)
    matrizes, targets = [], []
    for indice in indices:
        caso = casos[int(indice)]
        matrizes.append(construir_matriz_custom(
            caso["ranking_v2"], caso["features_por_dezena"],
            caso["extras_v5"], FEATURES_MULTIESCALA,
        ))
        targets.append([int(d in caso["nao_sorteadas"]) for d in range(1, 26)])
    X_meta = np.asarray(matrizes, dtype=np.float64)
    y_meta = np.asarray(targets, dtype=np.int8)
    if USAR_CACHE_META:
        cache.salvar_meta(indices, X_meta, y_meta)
    print(f"Matrizes meta criadas: {X_meta.shape}.")
    return indices, X_meta, y_meta


def probabilidades_modelo(modelo, X_alvo):
    classes = modelo.named_steps["logistic"].classes_
    coluna = int(np.where(classes == 1)[0][0])
    return modelo.predict_proba(X_alvo)[:, coluna]


def pesos_janelas(forcas: dict[int, float], janelas: list[int]) -> dict[int, float]:
    top = sorted(janelas, key=lambda w: (-forcas[w], w))[:min(SELETOR_TOP_K, len(janelas))]
    valores = np.asarray([forcas[w] for w in top], dtype=float)
    valores = np.exp((valores - valores.max()) / max(SELETOR_TEMPERATURA, 1e-9))
    valores = np.maximum(valores, SELETOR_PESO_MINIMO)
    valores /= valores.sum()
    return {w: float(p) for w, p in zip(top, valores)}


def logloss_binaria(y, p):
    p = np.clip(np.asarray(p), 1e-9, 1 - 1e-9)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def executar(casos, indices_meta, X_meta, y_meta, total):
    pos = {int(indice): i for i, indice in enumerate(indices_meta)}
    janelas = janelas_candidatas()
    if BASELINE_JANELA_FIXA not in janelas:
        raise ValueError("BASELINE_JANELA_FIXA precisa pertencer às janelas candidatas.")
    forcas = {w: float(SELETOR_PRIOR_FORCA) for w in janelas}
    inicio = max(min(pos) + max(janelas), total - BLOCOS_RELATORIO * TAMANHO_BLOCO_RELATORIO)
    detalhes, diagnostico_janelas, votos = [], [], []

    for ordem, indice in enumerate(range(inicio, total), 1):
        alvo = pos[indice]
        probs_por_janela = {}
        for w in janelas:
            modelo = treinar_modelo_meta(
                X_meta[alvo - w:alvo].reshape(-1, X_meta.shape[2]),
                y_meta[alvo - w:alvo].reshape(-1),
            )
            probs_por_janela[w] = probabilidades_modelo(modelo, X_meta[alvo])

        pesos = pesos_janelas(forcas, janelas)
        combinado = sum(pesos[w] * probs_por_janela[w] for w in pesos)
        anterior = casos[indice]["sorteadas_anterior"]
        ajuste = np.asarray([1.0 if d in anterior else 0.0 for d in range(1, 26)])
        score_final = combinado - (PESO_REPETICAO_ANTERIOR * ajuste if USAR_SINAL_REPETICAO else 0.0)
        ranking = np.argsort(-score_final) + 1
        ranking_fixo = np.argsort(-probs_por_janela[BASELINE_JANELA_FIXA]) + 1
        y_real = y_meta[alvo]
        baseline_loss = logloss_binaria(y_real, np.full(25, 0.4))

        for w in janelas:
            peso = pesos.get(w, 0.0)
            diagnostico_janelas.append({
                "indice": indice, "concurso": casos[indice]["concurso"],
                "janela": w, "forca_antes_alvo": forcas[w], "peso": peso,
                "selecionada": w in pesos,
            })
            ganho = baseline_loss - logloss_binaria(y_real, probs_por_janela[w])
            forcas[w] = (1 - SELETOR_ALPHA) * forcas[w] + SELETOR_ALPHA * ganho

        for d in range(1, 26):
            votos.append({
                "concurso": casos[indice]["concurso"], "dezena": d,
                "voto_ponderado": combinado[d - 1],
                "saiu_anterior": int(d in anterior), "ajuste_repeticao": -PESO_REPETICAO_ANTERIOR * int(d in anterior),
                "score_final": score_final[d - 1], "rank_final": int(np.where(ranking == d)[0][0] + 1),
            })

        for qtd in CENARIOS_EXCLUSOES:
            escolhidas = set(int(x) for x in ranking[:qtd])
            acertos = len(escolhidas & casos[indice]["nao_sorteadas"])
            base_v2 = {x["dezena"] for x in casos[indice]["ranking_v2"][:qtd]}
            base_fixa = set(int(x) for x in ranking_fixo[:qtd])
            detalhes.append({
                "indice": indice, "concurso": casos[indice]["concurso"],
                "qtd_exclusoes": qtd, "acertos": acertos,
                "taxa_acerto": acertos / qtd, "lift_vs_40_pct": 100 * ((acertos / qtd) / 0.4 - 1),
                "baseline_aleatorio": qtd * 0.4,
                "acertos_baseline_v2": len(base_v2 & casos[indice]["nao_sorteadas"]),
                "acertos_baseline_janela_fixa": len(base_fixa & casos[indice]["nao_sorteadas"]),
                "melhor_janela": max(pesos, key=pesos.get),
                "janelas_selecionadas": ",".join(map(str, pesos)),
            })
        if ordem % 10 == 0 or indice == total - 1:
            print(f"Rolling {ordem}/{total - inicio} | concurso {casos[indice]['concurso']}")
    return pd.DataFrame(detalhes), pd.DataFrame(diagnostico_janelas), pd.DataFrame(votos)


def adicionar_blocos(detalhes: pd.DataFrame) -> pd.DataFrame:
    saida = detalhes.copy()
    concursos = sorted(saida["concurso"].unique())
    mapa = {c: min(i // TAMANHO_BLOCO_RELATORIO + 1, BLOCOS_RELATORIO)
            for i, c in enumerate(concursos)}
    saida["bloco_relatorio"] = saida["concurso"].map(mapa)
    return saida


def resumos(detalhes):
    resumo = detalhes.groupby("qtd_exclusoes", as_index=False).agg(
        concursos=("concurso", "count"), media_acertos=("acertos", "mean"),
        taxa_acerto=("taxa_acerto", "mean"), baseline_v2=("acertos_baseline_v2", "mean"),
        baseline_janela_fixa=("acertos_baseline_janela_fixa", "mean"),
    )
    resumo["baseline_40"] = 0.4
    resumo["lift_vs_40_pct"] = 100 * (resumo["taxa_acerto"] / 0.4 - 1)
    resumo["ganho_vs_v2"] = resumo["media_acertos"] - resumo["baseline_v2"]
    resumo["ganho_vs_janela_fixa"] = resumo["media_acertos"] - resumo["baseline_janela_fixa"]
    blocos = detalhes.groupby(["bloco_relatorio", "qtd_exclusoes"], as_index=False).agg(
        concursos=("concurso", "count"), media_acertos=("acertos", "mean"),
        taxa_acerto=("taxa_acerto", "mean"), baseline_v2=("acertos_baseline_v2", "mean"),
        baseline_janela_fixa=("acertos_baseline_janela_fixa", "mean"),
    )
    blocos["lift_vs_40_pct"] = 100 * (blocos["taxa_acerto"] / 0.4 - 1)
    return resumo, blocos


def main():
    inicio = time.time()
    cache = criar_cache()
    casos, total = preparar_casos(cache)
    indices, X_meta, y_meta = preparar_matrizes_meta(casos, cache)
    detalhes, janelas, votos = executar(casos, indices, X_meta, y_meta, total)
    detalhes = adicionar_blocos(detalhes)
    resumo, blocos = resumos(detalhes)
    destino = ROOT / "experimentos" / ARQUIVO_SAIDA
    with pd.ExcelWriter(destino, engine="openpyxl") as writer:
        resumo.to_excel(writer, "Resumo", index=False)
        blocos.to_excel(writer, "Desempenho_Blocos", index=False)
        detalhes.to_excel(writer, "Detalhes", index=False)
        janelas.to_excel(writer, "Forca_Janelas", index=False)
        votos.to_excel(writer, "Votos_Ranking", index=False)
    print(resumo.round(4).to_string(index=False))
    print(f"Arquivo: {destino}\nTempo: {time.time() - inicio:.1f}s")


if __name__ == "__main__":
    main()
