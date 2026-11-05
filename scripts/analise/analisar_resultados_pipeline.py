#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisar_resultados_pipeline.py — Análise da execução completa da pipeline (zip baixado do Kaggle)
===================================================================================================

Reproduz, a partir das PREDIÇÕES gravadas (``runs/*.npz``), os números da execução completa do notebook (100 treinos =
5 folds × 5 seeds × 4 variantes) e acrescenta o que as tabelas do notebook não trazem. É INDEPENDENTE do notebook: só lê
arquivos (o notebook baixado é lido apenas como JSON, nunca executado; os .npz são lidos com ``allow_pickle=False``) e
reaproveita as funções de janelas do ``estatisticas_janelas.py``.

O QUE FAZ
---------
 1. Confere a execução: 100 treinos, assinatura única; ambiente, duplicatas, tempo e diagnósticos da KAN vêm do log do notebook.
 2. Reconstrói as janelas (65.670) e VALIDA o mapeamento ``idx_teste`` → (ciclone, instante): os rótulos e o fold de cada
    uma das 100 execuções precisam coincidir com os reconstruídos (senão o script aborta: sem isso nada abaixo vale).
 3. Recalcula as métricas de cada execução e confere com ``tabelas/metricas_por_execucao.csv``; decompõe a variação em
    entre folds × entre seeds.
 4. Refaz o bootstrap por ciclone do notebook (confere com ``tabelas/bootstrap_por_ciclone.csv``).
 5. Faixa horizontal do gráfico predito × real: quais janelas são, qual a causa e quanto pesam. Achado: longitude do ciclone
    201407E com o sinal trocado depois da linha de data (o recorte fica centrado no ponto espelhado e o ciclone sai do
    quadro). Sensibilidade: as mesmas métricas e os mesmos testes sem as janelas afetadas.
 6. ΔVmax e intensificação rápida (RI): reproduz o baseline de histórico, faz bootstrap por ciclone PAREADO de AUC e AP e
    um empilhamento (a rede acrescenta algo ao histórico de Vmax?; estimativa exploratória, não substitui um treino).
 7. Sobreajuste (curvas gravadas), erro por faixa de intensidade, janelas de 2017 (referência ao TCC).

USO
---
    python3 -I scripts/analise/analisar_resultados_pipeline.py                      # usa os padrões abaixo
    python3 -I scripts/analise/analisar_resultados_pipeline.py --entrada resultados_pipeline/saida_pipeline --dados data \\
            --saida resultados_analise_pipeline --reps-ri 2000

``-I`` (modo isolado): o zip baixado é dado não confiável; assim o Python não importa nada da pasta dele.

SAÍDAS (em ``resultados_analise_pipeline/``)
--------------------------------------------
    resumo_analise.json      todos os números (para conferir/citar)
    relatorio_analise.md     relatório legível, gerado a partir dos números
    tabelas/*.csv            uma tabela por análise
    figuras/*.png            faixa antes/depois, quadros do 201407E, trilha e erro, candidatos da linha de data
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps

RAIZ = Path(__file__).resolve().parents[2]  # raiz do projeto (este arquivo fica em scripts/analise/)
sys.path.insert(0, str(RAIZ / "scripts" / "dados"))  # com `python3 -I` a pasta do script não entra sozinha em sys.path
import estatisticas_janelas as ej           # noqa: E402  (mesmas regras de janelas do notebook)

# ----------------------------------------------------------------------------- parâmetros
SEED_ESTRUTURAL = ej.SEED_ESTRUTURAL        # 2024: folds e bootstrap (cfg.seed_estrutural do notebook)
LIMIAR_RI = ej.LIMIAR_RI_NOS                # 30 nós
PX_GRAUS = 0.07                             # tamanho de um pixel do TCIR (≈ 7,8 km)
# Ciclone com a longitude de ordem trocada depois da linha de data (ver `analisar_faixa`) e o 1º quadro afetado: o ponto de
# virada da longitude (máximo de |lon|, 07/ago 06h) é o último quadro com a longitude certa.
CICLONE_LON_TROCADA = ("201407E", pd.Timestamp("2014-08-07 09:00"))
LIMITE_FAIXA_REAL, LIMITE_FAIXA_PREV = 60.0, 35.0     # "faixa": Vmax real ≥ 60 e previsto ≤ 35 nós
BINS_INTENSIDADE = [(10, 34, "DT (< 34)"), (34, 64, "TS (34–63)"), (64, 83, "Cat 1 (64–82)"), (83, 96, "Cat 2 (83–95)"),
                    (96, 113, "Cat 3 (96–112)"), (113, 137, "Cat 4 (113–136)"), (137, 1000, "Cat 5 (≥ 137)")]
TCC_REF = {"CNN-TC (TCC)": (10.06, 7.53), "MobileNetV2+SDTA (TCC)": (9.14, 7.00)}     # RMSE, MAE no teste de 2017 do TCC


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "sim" if x else "não"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def tabela_md(df: pd.DataFrame, nd: int = 3) -> str:
    """Tabela em Markdown (sem depender de `tabulate`)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.astype(object).iterrows():             # astype(object): inteiros continuam inteiros
        out.append("| " + " | ".join(fmt(x, nd) for x in r) + " |")
    return "\n".join(out)


def json_seguro(o):
    if isinstance(o, dict):
        return {str(k): json_seguro(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_seguro(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


# ----------------------------------------------------------------------------- 1. execuções e log do notebook
def carregar_execucoes(dir_runs: Path) -> list:
    execs = []
    for p in sorted(dir_runs.glob("*.npz")):
        with np.load(p, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"]))
            execs.append(dict(
                variante=meta["variante"], fold=int(meta["fold"]), seed=int(meta["seed"]), meta=meta, arquivo=p.name,
                assinatura=p.stem.split("__")[-1], idx_teste=d["idx_teste"].astype(np.int64),
                y_vmax=d["y_vmax"].astype(np.float64), pred_vmax=d["pred_vmax"].astype(np.float64),
                y_dvmax=d["y_dvmax"].astype(np.float64), pred_dvmax=d["pred_dvmax"].astype(np.float64),
                y_dvmax_val=d["y_dvmax_val"].astype(np.float64), pred_dvmax_val=d["pred_dvmax_val"].astype(np.float64)))
    return execs


def ler_log_notebook(caminho: Path) -> str:
    """Texto de saída das células de código (o .ipynb baixado é lido só como JSON; nunca é executado)."""
    nb = json.loads(caminho.read_text(encoding="utf-8"))
    partes = []
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                partes.append("".join(o.get("text", [])))
            elif o.get("output_type") in ("execute_result", "display_data"):
                t = o.get("data", {}).get("text/plain")
                if t:
                    partes.append("".join(t))
    return "\n".join(partes)


RE_RUN = re.compile(r"\[(\d\d:\d\d:\d\d)\] \[(\d+)/(\d+)\] (\w+)\s+fold (\d+) seed (\d+) \| Vmax RMSE ([\d.]+) MAE ([\d.]+) nós"
                    r"(?: \| ΔVmax RMSE ([\d.]+))? \| melhor época (\d+)/(\d+) \| (?:(\d+)h)?(\d+)m(\d+)s")


def segundos_do_dia(hms: str) -> int:
    h, m, s = map(int, hms.split(":"))
    return 3600 * h + 60 * m + s


def analisar_log(texto: str) -> dict:
    r = {}
    m = re.search(r"python (\S+) \| torch (\S+) \| numpy (\S+) \| pandas (\S+) \| h5py (\S+)", texto)
    if m:
        r["ambiente"] = dict(python=m[1], torch=m[2], numpy=m[3], pandas=m[4], h5py=m[5])
    m = re.search(r"GPU: (.+?) \| ([\d.]+) GB", texto)
    if m:
        r["gpu"] = dict(nome=m[1], memoria_gb=float(m[2]))
    m = re.search(r"duplicatas: (\d+) registros a remover, (\d+) ID\(s\) unificado\(s\)", texto)
    if m:
        r["duplicatas"] = dict(registros_removidos=int(m[1]), ids_unificados=int(m[2]))
    m = re.search(r"(\d+) de (\d+) registros removidos são idênticos bit a bit", texto)
    if m:
        r.setdefault("duplicatas", {})["identicos_bit_a_bit"] = f"{m[1]} de {m[2]}"
    m = re.search(r"janelas: (\d+) de (\d+) ciclones \| com rótulo ΔVmax: (\d+) \| RI \(ΔVmax ≥ 30 nós\): (\d+)", texto)
    if m:
        r["janelas_log"] = dict(janelas=int(m[1]), ciclones=int(m[2]), com_dvmax=int(m[3]), ri=int(m[4]))
    m = re.search(r"quadros na série: (\d+) \(inválidos: (\d+)", texto)
    if m:
        r["quadros_invalidos"] = int(m[2])
    t0 = re.search(r"\[(\d\d:\d\d:\d\d)\] plano: (\d+) execuções", texto)
    t1 = re.search(r"\[(\d\d:\d\d:\d\d)\] protocolo concluído", texto)
    if t0 and t1:
        dur = (segundos_do_dia(t1[1]) - segundos_do_dia(t0[1])) % 86400            # a sessão atravessa a meia-noite
        r["tempo_treino"] = dict(inicio=t0[1], fim=t1[1], segundos=dur, horas=round(dur / 3600, 2), plano=int(t0[2]))
    runs = []
    for m in RE_RUN.finditer(texto):
        runs.append(dict(ordem=int(m[2]), variante=m[4], fold=int(m[5]), seed=int(m[6]), rmse_vmax=float(m[7]), mae_vmax=float(m[8]),
                         rmse_dvmax=float(m[9]) if m[9] else np.nan, melhor_epoca=int(m[10]), n_epocas=int(m[11]),
                         segundos=3600 * int(m[12] or 0) + 60 * int(m[13]) + int(m[14])))
    r["runs_log"] = pd.DataFrame(runs)
    kan = {}
    for m in re.finditer(r"KAN \((vmax|dvmax)\): as 10 variáveis mais importantes concentram ([\d.]+)% da importância "
                         r"\(se fosse uniforme: ([\d.]+)%\) \| nº efetivo de variáveis = (\d+) de (\d+)", texto):
        kan[m[1]] = dict(top10_pct=float(m[2]), uniforme_pct=float(m[3]), n_efetivo=int(m[4]), n_total=int(m[5]))
    m = re.search(r"variáveis latentes entre as 10 mais importantes nas duas tarefas: \[(.*?)\] \((\d+)/10 em comum; "
                  r"por acaso esperaríamos ([\d.]+), P\(≥ observado\) = ([\d.]+)\)", texto)
    if m:
        kan["sobreposicao"] = dict(comuns=[int(x) for x in m[1].split(",") if x.strip()], n=int(m[2]), esperado_acaso=float(m[3]), p=float(m[4]))
    r["kan"] = kan
    m = re.search(r"(\d+) de (\d+) janelas \(([\d.]+)%\) têm \|erro de Vmax\| ≥ (\d+) nós em (\w+); elas respondem por ([\d.]+)% da soma dos "
                  r"erros quadráticos \(RMSE sem elas: ([\d.]+) nós\)", texto)
    if m:
        r["erro_grande_log"] = dict(n=int(m[1]), de=int(m[2]), limite=int(m[4]), pct_sq=float(m[6]), rmse_sem=float(m[7]))
    return r


# ----------------------------------------------------------------------------- 2. janelas e validação do mapeamento
def reconstruir_janelas(dir_dados: Path, cache: Path):
    arqs = ej.localizar_arquivos(dir_dados)
    tabela = ej.carregar_tabela(arqs)
    frac = ej.obter_frac_invalido_ir(tabela, arqs, cache, refazer=False)
    jan, serie = ej.construir_janelas(tabela, frac)
    fold = ej.dividir_folds(jan["id_ciclone"].to_numpy(), ej.N_FOLDS, SEED_ESTRUTURAL)
    serie = serie.copy()
    serie["lon_n"] = ((serie["lon"].to_numpy(float) + 180.0) % 360.0) - 180.0            # o CPAC usa 0…360
    chave = pd.MultiIndex.from_arrays([jan["id_ciclone"].to_numpy(), jan["instante"].to_numpy()])
    s = serie.set_index(["id_ciclone", "instante"])
    jan = jan.copy()
    jan["lon_t"] = s["lon_n"].reindex(chave).to_numpy()
    jan["lat_t"] = s["lat"].reindex(chave).to_numpy()
    jan["dist_linha_data"] = 180.0 - np.abs(jan["lon_t"].to_numpy())
    return jan, serie, fold, arqs


def validar_mapeamento(execs: list, jan: pd.DataFrame, fold: np.ndarray) -> dict:
    """Cada .npz guarda os ÍNDICES das janelas de teste. Se a reconstrução local das janelas não for idêntica à do notebook, os
    índices apontam para outras janelas e toda análise por ciclone/posição seria falsa: aqui isso é uma falha, não um aviso."""
    y, dv = jan["vmax"].to_numpy(np.float32), jan["dvmax"].to_numpy(np.float32)
    n = len(jan)
    ruins = []
    for e in execs:
        i = e["idx_teste"]
        ok = (i.min() >= 0 and i.max() < n
              and np.array_equal(y[i].astype(np.float64), e["y_vmax"])
              and np.array_equal(dv[i].astype(np.float64), e["y_dvmax"], equal_nan=True)
              and np.array_equal(np.sort(i), np.flatnonzero(fold == e["fold"])))
        if not ok:
            ruins.append(e["arquivo"])
    if ruins:
        raise SystemExit(f"ERRO: {len(ruins)} execução(ões) não batem com as janelas reconstruídas (p. ex. {ruins[:3]}). "
                         "Confira os dados (data/), o cache frac_invalido_ir.csv e as regras de janelas.")
    return dict(execucoes_conferidas=len(execs), janelas=n, ciclones=int(jan["id_ciclone"].nunique()))


def matrizes_previsao(execs: list, variantes: list, seeds: list, n_jan: int) -> dict:
    P = {v: {t: np.full((len(seeds), n_jan), np.nan) for t in ("vmax", "dvmax")} for v in variantes}
    for e in execs:
        i = seeds.index(e["seed"])
        P[e["variante"]]["vmax"][i, e["idx_teste"]] = e["pred_vmax"]
        P[e["variante"]]["dvmax"][i, e["idx_teste"]] = e["pred_dvmax"]
    return P


def erros_medios(P: np.ndarray, y: np.ndarray):
    """|erro| e erro² de cada janela, médios entre as seeds (como no notebook); NaN onde não há rótulo ou previsão."""
    err = P - y[None, :]
    cnt = (~np.isnan(err)).sum(0)
    ab = np.where(cnt > 0, np.nansum(np.abs(err), 0) / np.maximum(cnt, 1), np.nan)
    sq = np.where(cnt > 0, np.nansum(err ** 2, 0) / np.maximum(cnt, 1), np.nan)
    return ab, sq


# ----------------------------------------------------------------------------- 3. métricas (cópias das funções do notebook)
def metricas_regressao(y: np.ndarray, p: np.ndarray) -> dict:
    ok = ~(np.isnan(y) | np.isnan(p))
    if not ok.any():
        return dict(rmse=np.nan, mae=np.nan, vies=np.nan, n=0)
    e = p[ok] - y[ok]
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), vies=float(np.mean(e)), n=int(ok.sum()))


def metricas_ri(y_dv, p_dv, limiar, limiar_prev=None) -> dict:
    ok = ~(np.isnan(y_dv) | np.isnan(p_dv))
    real, prev = y_dv[ok] >= limiar, p_dv[ok] >= (limiar if limiar_prev is None else limiar_prev)
    tp, fn, fp = int((real & prev).sum()), int((real & ~prev).sum()), int((~real & prev).sum())
    div = lambda a, b: float(a / b) if b > 0 else np.nan
    return dict(pod=div(tp, tp + fn), far=div(fp, tp + fp), csi=div(tp, tp + fn + fp))


def auc_roc(real: np.ndarray, escore: np.ndarray) -> float:
    n1, n0 = int(real.sum()), int((~real).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = sps.rankdata(escore)
    return float((r[real].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def precisao_media(real: np.ndarray, escore: np.ndarray) -> float:
    if real.sum() == 0:
        return np.nan
    r = real[np.argsort(-escore, kind="stable")]
    acertos = np.cumsum(r)
    return float((acertos / np.arange(1, len(r) + 1))[r].sum() / r.sum())


def limiar_ri_ajustado(y_val, p_val, limiar_ri) -> float:
    ok = ~(np.isnan(y_val) | np.isnan(p_val))
    if not (y_val[ok] >= limiar_ri).any():
        return float(limiar_ri)
    grade = np.arange(min(5.0, limiar_ri), limiar_ri + 0.5, 1.0)
    csi = [metricas_ri(y_val, p_val, limiar_ri, limiar_prev=g)["csi"] for g in grade]
    return float(grade[int(np.argmax(np.nan_to_num(csi, nan=-1.0)))])


def metricas_por_execucao(execs: list) -> pd.DataFrame:
    linhas = []
    for e in execs:
        mv, md = metricas_regressao(e["y_vmax"], e["pred_vmax"]), metricas_regressao(e["y_dvmax"], e["pred_dvmax"])
        linha = dict(variante=e["variante"], fold=e["fold"], seed=e["seed"], rmse_vmax=mv["rmse"], mae_vmax=mv["mae"], vies_vmax=mv["vies"],
                     rmse_dvmax=md["rmse"], mae_dvmax=md["mae"], auc_ri=np.nan, ap_ri=np.nan, csi_ri_aj=np.nan,
                     melhor_epoca=e["meta"]["melhor_epoca"] + 1, n_epocas=e["meta"]["n_epocas"], segundos=e["meta"]["segundos"],
                     rmse_val_interna=e["meta"]["rmse_val"])
        if md["n"] > 0:
            ok = ~(np.isnan(e["y_dvmax"]) | np.isnan(e["pred_dvmax"]))
            real = e["y_dvmax"][ok] >= LIMIAR_RI
            lim = limiar_ri_ajustado(e["y_dvmax_val"], e["pred_dvmax_val"], LIMIAR_RI)
            linha.update(auc_ri=auc_roc(real, e["pred_dvmax"][ok]), ap_ri=precisao_media(real, e["pred_dvmax"][ok]),
                         csi_ri_aj=metricas_ri(e["y_dvmax"], e["pred_dvmax"], LIMIAR_RI, limiar_prev=lim)["csi"])
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values(["variante", "fold", "seed"]).reset_index(drop=True)


def conferir_com_notebook(df: pd.DataFrame, caminho_csv: Path) -> dict:
    nb = pd.read_csv(caminho_csv)
    m = df.merge(nb, on=["variante", "fold", "seed"], suffixes=("", "_nb"))
    out = dict(linhas_comparadas=int(len(m)))
    for c in ("rmse_vmax", "mae_vmax", "vies_vmax", "rmse_dvmax", "mae_dvmax", "auc_ri", "ap_ri", "csi_ri_aj"):
        d = (m[c] - m[c + "_nb"]).abs()
        out[c] = float(d.max()) if d.notna().any() else None
    return out


def decompor_variacao(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE de Vmax: variação ENTRE FOLDS (dp das 5 médias por fold) × ENTRE SEEDS (dp médio dentro de cada fold)."""
    linhas = []
    for v, g in df.groupby("variante"):
        R = g.pivot(index="fold", columns="seed", values="rmse_vmax")
        linhas.append(dict(variante=v, rmse_medio=R.to_numpy().mean(), entre_folds=float(R.mean(axis=1).std(ddof=1)),
                           entre_seeds=float(R.std(axis=1, ddof=1).mean()), **{f"fold_{f}": float(R.loc[f].mean()) for f in R.index}))
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------- 4. bootstrap por ciclone (réplica do notebook)
def holm_bonferroni(p: np.ndarray, alfa: float = 0.05):
    p = np.asarray(p, dtype=float)
    m = len(p)
    ordem = np.argsort(p)
    ajustado, maximo = np.empty(m), 0.0
    for posto, i in enumerate(ordem):
        maximo = max(maximo, (m - posto) * p[i])
        ajustado[i] = min(1.0, maximo)
    return ajustado <= alfa, ajustado


def bootstrap_por_ciclone(erros: dict, ref: str, outras: list, ids_ciclone: np.ndarray, reps: int, seed: int,
                          alfa: float = 0.05, mascara=None) -> pd.DataFrame:
    """Mesma conta do `comparar_bootstrap` do notebook (mesma semente ⇒ mesmos sorteios de ciclones). `mascara` (opcional) tira janelas
    da comparação — usada na análise de sensibilidade. `erros[v][tarefa] = (|erro|, erro²)` por janela, médios entre seeds."""
    rng = np.random.default_rng(seed)
    ids, inv = np.unique(ids_ciclone, return_inverse=True)
    n_c = len(ids)
    pesos = rng.multinomial(n_c, np.full(n_c, 1.0 / n_c), size=reps).astype(float)
    linhas = []
    for outra in outras:
        for t in ("vmax", "dvmax"):
            a_ref, q_ref = erros[ref][t]
            a_out, q_out = erros[outra][t]
            ok = ~(np.isnan(a_ref) | np.isnan(a_out))
            if mascara is not None:
                ok &= mascara
            if ok.sum() < 20:
                continue
            S = np.column_stack([np.bincount(inv[ok], weights=v, minlength=n_c) for v in
                                 (np.ones(int(ok.sum())), a_ref[ok], a_out[ok], q_ref[ok], q_out[ok])])
            difs = lambda T: dict(mae=(T[..., 1] - T[..., 2]) / T[..., 0], rmse=np.sqrt(T[..., 3] / T[..., 0]) - np.sqrt(T[..., 4] / T[..., 0]))
            obs, bs = difs(S.sum(0)), difs(pesos @ S)
            for m in ("mae", "rmse"):
                d = bs[m]
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = min(1.0, 2.0 * (min(int((d >= 0).sum()), int((d <= 0).sum())) + 1) / (reps + 1))
                linhas.append(dict(comparacao=f"{ref} − {outra}", tarefa=t, metrica=m, ciclones=int(len(np.unique(inv[ok]))), janelas=int(ok.sum()),
                                   dif=float(obs[m]), ic95_inf=float(lo), ic95_sup=float(hi), p_boot=float(p)))
    df = pd.DataFrame(linhas)
    df["significativo"], df["p_holm"] = holm_bonferroni(df["p_boot"].to_numpy(), alfa)
    df["veredito"] = np.where(~df["significativo"], "sem diferença", np.where(df["dif"] < 0, "referência melhor", "referência pior"))
    return df


# ----------------------------------------------------------------------------- 5. baseline de histórico (réplica do notebook)
def features_historico(jan: pd.DataFrame, serie: pd.DataFrame) -> np.ndarray:
    s = serie.set_index(["id_ciclone", "instante"])["vmax"]
    v0 = jan["vmax"].to_numpy(float)
    cols = [v0]
    for h in (3, 6, 12):
        idx = pd.MultiIndex.from_arrays([jan["id_ciclone"].to_numpy(), (jan["instante"] - pd.to_timedelta(h, unit="h")).to_numpy()])
        cols.append(v0 - s.reindex(idx).to_numpy(float))
    return np.column_stack(cols)


def polinomio_grau2(Z: np.ndarray) -> np.ndarray:
    p = Z.shape[1]
    return np.column_stack([np.ones(len(Z))] + [Z[:, i] for i in range(p)] + [Z[:, i] * Z[:, j] for i in range(p) for j in range(i, p)])


def baseline_historico(jan: pd.DataFrame, serie: pd.DataFrame, fold: np.ndarray, p_rede: np.ndarray = None):
    """Previsão FORA DA AMOSTRA do baseline de histórico (regressão polinomial de grau 2 em Vmax(t) e na tendência de 3, 6 e 12 h), uma
    regressão por fold ajustada nos outros folds. Se `p_rede` (ΔVmax previsto pela rede, fora da amostra) for dado, devolve também o
    EMPILHAMENTO: a mesma regressão com a previsão da rede como variável extra."""
    X = features_historico(jan, serie)
    y = jan["dvmax"].to_numpy(float)
    x_ok = ~np.isnan(X).any(1)
    prev = np.full(len(jan), np.nan)
    prev_emp = np.full(len(jan), np.nan)
    for f in range(ej.N_FOLDS):
        tr = x_ok & ~np.isnan(y) & (fold != f)
        te = np.flatnonzero(fold == f)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Phi = polinomio_grau2((X - mu) / sd)
        w, *_ = np.linalg.lstsq(Phi[tr], y[tr], rcond=None)
        prev[te] = (Phi @ w)[te]
        if p_rede is not None:
            ok_r = tr & ~np.isnan(p_rede)
            mu_r, sd_r = p_rede[ok_r].mean(), p_rede[ok_r].std() + 1e-9
            Phi2 = np.column_stack([Phi, (p_rede - mu_r) / sd_r])
            w2, *_ = np.linalg.lstsq(Phi2[ok_r], y[ok_r], rcond=None)
            prev_emp[te] = (Phi2 @ w2)[te]
    return prev, prev_emp


# ----------------------------------------------------------------------------- 6. RI: bootstrap por ciclone pareado
def bootstrap_ri(jan: pd.DataFrame, scores: dict, reps: int, seed: int) -> pd.DataFrame:
    """AUC e AP (pooled nos 5 folds) de cada escore de ΔVmax como previsor de RI, e diferenças pareadas entre escores com IC 95% por
    reamostragem de ciclones (as mesmas janelas em todos os escores). Escores: ΔVmax previsto (rede ou baseline)."""
    dv = jan["dvmax"].to_numpy(float)
    ok = ~np.isnan(dv)
    idx = np.flatnonzero(ok)
    real = dv[idx] >= LIMIAR_RI
    ids, inv_all = np.unique(jan["id_ciclone"].to_numpy(), return_inverse=True)
    n_c = len(ids)
    cic = inv_all[idx]
    ordem = np.argsort(cic, kind="stable")
    cortes = np.searchsorted(cic[ordem], np.arange(n_c + 1))
    por_ciclone = [ordem[cortes[c]:cortes[c + 1]] for c in range(n_c)]
    S = {k: v[idx] for k, v in scores.items()}
    nomes = list(S)
    obs = {k: (auc_roc(real, S[k]), precisao_media(real, S[k])) for k in nomes}
    rng = np.random.default_rng(seed)
    auc_b = {k: np.empty(reps) for k in nomes}
    ap_b = {k: np.empty(reps) for k in nomes}
    for r in range(reps):
        amostra = rng.integers(0, n_c, n_c)
        sel = np.concatenate([por_ciclone[c] for c in amostra if len(por_ciclone[c])])
        re_ = real[sel]
        for k in nomes:
            auc_b[k][r] = auc_roc(re_, S[k][sel])
            ap_b[k][r] = precisao_media(re_, S[k][sel])
    linhas = []
    ref = nomes[0]
    for k in nomes:
        linhas.append(dict(escore=k, auc=obs[k][0], auc_ic_inf=float(np.percentile(auc_b[k], 2.5)), auc_ic_sup=float(np.percentile(auc_b[k], 97.5)),
                           ap=obs[k][1], ap_ic_inf=float(np.percentile(ap_b[k], 2.5)), ap_ic_sup=float(np.percentile(ap_b[k], 97.5)),
                           dif_auc_vs_ref=np.nan, dif_auc_ic_inf=np.nan, dif_auc_ic_sup=np.nan, p_auc=np.nan,
                           dif_ap_vs_ref=np.nan, dif_ap_ic_inf=np.nan, dif_ap_ic_sup=np.nan, p_ap=np.nan))
        if k == ref:
            continue
        for nome, b, o in (("auc", auc_b, obs), ("ap", ap_b, obs)):
            d = b[ref] - b[k]
            col = 0 if nome == "auc" else 1
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = min(1.0, 2.0 * (min(int((d >= 0).sum()), int((d <= 0).sum())) + 1) / (reps + 1))
            linhas[-1].update({f"dif_{nome}_vs_ref": o[ref][col] - o[k][col], f"dif_{nome}_ic_inf": float(lo), f"dif_{nome}_ic_sup": float(hi), f"p_{nome}": float(p)})
    df = pd.DataFrame(linhas)
    # Holm sobre TODOS os testes desta tabela (cada escore ≠ referência × {AUC, AP})
    p_todos = np.concatenate([df["p_auc"].to_numpy()[1:], df["p_ap"].to_numpy()[1:]])
    _, p_adj = holm_bonferroni(p_todos)
    df["p_auc_holm"] = np.concatenate([[np.nan], p_adj[:len(df) - 1]])
    df["p_ap_holm"] = np.concatenate([[np.nan], p_adj[len(df) - 1:]])
    return df


# ----------------------------------------------------------------------------- 7. a faixa horizontal e a longitude trocada
def detectar_reflexao(serie: pd.DataFrame, topo_min: float = 178.0, passo_min: float = 0.3, viz: int = 3) -> pd.DataFrame:
    """Triagem (NÃO é prova): ciclones cuja |lon| sobe até um máximo ≥ 178° e desce logo depois SEM troca de sinal, com passo médio ≥ 0,3°
    por quadro dos dois lados. É a assinatura de uma longitude 'refletida' na linha de data (o ciclone segue para oeste, mas a longitude
    gravada volta). Um ciclone de verdade que desacelera e volta a andar não passa: o passo cai a ~0 antes de inverter."""
    linhas = []
    for id_c, g in serie.sort_values(["id_ciclone", "instante"]).groupby("id_ciclone"):
        lon = g["lon_n"].to_numpy()
        a = np.abs(lon)
        for i in range(viz, len(a) - viz):
            if a[i] < topo_min or (i > viz and a[i - 1] == a[i]):
                continue
            w = slice(i - viz, i + viz + 1)
            if a[i] < a[w].max() or np.unique(np.sign(lon[w])).size > 1:
                continue
            sobe, desce = np.diff(a[i - viz:i + 1]).mean(), np.diff(a[i:i + viz + 1]).mean()
            if sobe >= passo_min and desce <= -passo_min:
                linhas.append(dict(id_ciclone=id_c, instante_topo=g["instante"].iloc[i], lon_topo=float(lon[i]), lat=float(g["lat"].iloc[i]),
                                   vmax_topo=float(g["vmax"].iloc[i]), passo_antes=float(sobe), passo_depois=float(desce), n_quadros=int(len(g))))
    return pd.DataFrame(linhas)


def ler_ir(arqs: dict, r) -> np.ndarray:
    with h5py.File(arqs[r["arquivo"]], "r") as h:
        ir = h["matrix"][int(r["linha"])][..., 0].astype(np.float32)
    return np.where(~np.isfinite(ir) | (ir > 1000), np.nan, ir)


def figura_quadros(serie: pd.DataFrame, arqs: dict, id_c: str, instantes: list, t_troca, caminho: Path) -> None:
    """IR 201×201 do ciclone: '+' = centro do recorte; '×' = onde o ciclone estaria se a longitude gravada tivesse o sinal trocado, isto é,
    2·(180 − |lon|)° a OESTE do centro. Antes do ponto de troca, a longitude está certa e o ciclone aparece no '+'."""
    s = serie.set_index(["id_ciclone", "instante"])
    n = len(instantes)
    ncol = 6
    fig, ax = plt.subplots((n + ncol - 1) // ncol, ncol, figsize=(3.1 * ncol, 3.5 * ((n + ncol - 1) // ncol)), squeeze=False,
                           constrained_layout=True)
    for a, t in zip(ax.ravel(), instantes):
        r = s.loc[(id_c, pd.Timestamp(t))]
        a.imshow(ler_ir(arqs, r), cmap="gray_r")
        a.plot([100], [100], "+", c="yellow", ms=11, mew=1.5)
        a.add_patch(plt.Rectangle((54, 54), 92, 92, fill=False, ec="cyan", lw=1))
        depois = pd.Timestamp(t) >= t_troca
        if depois:
            desloc = 2 * (180 - abs(r["lon_n"])) / PX_GRAUS
            a.plot([100 - desloc], [100], "x", c="red", ms=10, mew=2)
            rot = f"× a {desloc:.0f} px"
        else:
            rot = "longitude certa"
        a.set_xlim(0, 200)
        a.set_ylim(200, 0)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"{pd.Timestamp(t):%d/%m %Hh}  lon {r['lon_n']:.1f}°  Vmax {r['vmax']:.0f}\n{rot}", fontsize=8)
    for a in ax.ravel()[n:]:
        a.axis("off")
    fig.suptitle(f"{id_c} — IR 201×201 (+ centro do recorte, retângulo ciano = recorte 92×92 usado pela rede, × = ciclone se a longitude "
                 "tiver o sinal trocado)", fontsize=10)
    plt.savefig(caminho, dpi=85)
    plt.close(fig)


def figura_candidatos(serie: pd.DataFrame, arqs: dict, cand: pd.DataFrame, caminho: Path) -> None:
    s = serie.set_index(["id_ciclone", "instante"])
    offs = (-4, 0, 2, 4, 6)
    fig, ax = plt.subplots(len(cand), len(offs), figsize=(3.0 * len(offs), 3.1 * len(cand)), squeeze=False)
    for k, (_, c) in enumerate(cand.iterrows()):
        g = serie[serie["id_ciclone"] == c["id_ciclone"]].sort_values("instante").reset_index(drop=True)
        i0 = int(np.flatnonzero(g["instante"].to_numpy() == np.datetime64(c["instante_topo"]))[0])
        for j, off in enumerate(offs):
            a = ax[k, j]
            r = g.iloc[min(max(i0 + off, 0), len(g) - 1)]
            a.imshow(ler_ir(arqs, r), cmap="gray_r")
            desloc = 2 * (180 - abs(r["lon_n"])) / PX_GRAUS
            a.plot([100], [100], "+", c="yellow", ms=10, mew=1.5)
            a.plot([100 - desloc], [100], "x", c="red", ms=9, mew=2)
            a.set_xlim(0, 200)
            a.set_ylim(200, 0)
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"{c['id_ciclone']} {r['instante']:%d/%m/%y %Hh} lon {r['lon_n']:.1f}\nVmax {r['vmax']:.0f} | × a {desloc:.0f} px", fontsize=8)
    fig.suptitle("Candidatos a longitude trocada (triagem pela trilha): o ciclone está no + ou no ×?", fontsize=10, y=0.998)
    plt.tight_layout()
    plt.savefig(caminho, dpi=75)
    plt.close(fig)


def figura_trilha(jan: pd.DataFrame, pred: np.ndarray, id_c: str, t_troca, caminho: Path) -> pd.DataFrame:
    """Trilha (longitude gravada × longitude se o sinal estivesse trocado) e Vmax real × previsto, no tempo."""
    m = (jan["id_ciclone"] == id_c).to_numpy()
    j = jan.loc[m, ["instante", "vmax", "lon_t", "lat_t"]].copy()
    j["pred"] = pred[m]
    j["lon_se_trocada"] = np.where(j["instante"] >= t_troca, -360.0 - j["lon_t"], j["lon_t"])      # reflexão em torno de −180°
    j["desloc_prev_px"] = np.where(j["instante"] >= t_troca, 2 * (180 - j["lon_t"].abs()) / PX_GRAUS, 0.0)
    j = j[(j["instante"] >= "2014-08-05") & (j["instante"] <= "2014-08-12")] if id_c == CICLONE_LON_TROCADA[0] else j
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(j["instante"], j["lon_t"], "o-", ms=3, label="longitude gravada no TCIR")
    ax[0].plot(j["instante"], j["lon_se_trocada"], "s--", ms=3, c="tab:red", label="longitude se o sinal estivesse trocado depois da virada")
    ax[0].axhline(-180, c="gray", lw=0.8)
    ax[0].axvline(t_troca, c="gray", ls=":")
    ax[0].set_ylabel("longitude (°)")
    ax[0].legend(fontsize=8)
    ax[1].plot(j["instante"], j["vmax"], "o-", ms=3, label="Vmax real")
    ax[1].plot(j["instante"], j["pred"], "o-", ms=3, c="tab:orange", label="previsto (cnn_lstm_kan, média das seeds)")
    ax[1].axvline(t_troca, c="gray", ls=":")
    ax[1].set_ylabel("Vmax (nós)")
    ax[1].legend(fontsize=8)
    ax[2].plot(j["instante"], j["desloc_prev_px"], "o-", ms=3, c="tab:red")
    ax[2].axhline(46, c="cyan", ls="--", lw=1, label="meia largura do recorte 92×92 (46 px)")
    ax[2].axhline(100, c="blue", ls="--", lw=1, label="meia largura do quadro 201×201 (100 px)")
    ax[2].axvline(t_troca, c="gray", ls=":")
    ax[2].set_ylabel("deslocamento do ciclone\nem relação ao centro (px)")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"{id_c}: a longitude gravada volta a crescer depois de −179,8° e o erro cresce junto com o deslocamento previsto")
    plt.tight_layout()
    plt.savefig(caminho, dpi=90)
    plt.close(fig)
    return j


def figura_faixa(y: np.ndarray, pred: np.ndarray, afetada: np.ndarray, caminho: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    for a, m, tit in ((ax[0], np.ones(len(y), bool), "todas as janelas"), (ax[1], ~afetada, "sem as janelas afetadas do 201407E")):
        h = a.hexbin(y[m], pred[m], gridsize=45, bins="log", mincnt=1, cmap="viridis", extent=(10, 190, 10, 190))
        a.plot([10, 190], [10, 190], "r--", lw=1)
        a.set_title(f"cnn_lstm_kan — {tit}\nRMSE {np.sqrt(np.mean((pred[m] - y[m]) ** 2)):.2f} nós | N = {int(m.sum())}", fontsize=10)
        a.set_xlabel("Vmax real (nós)")
    ax[0].scatter(y[afetada], pred[afetada], s=9, c="red", label=f"201407E depois da troca (n = {int(afetada.sum())})")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_ylabel("Vmax previsto (nós, média das seeds)")
    plt.tight_layout()
    plt.savefig(caminho, dpi=90)
    plt.close(fig)


def analisar_faixa(jan, serie, arqs, y, pred, afetada, saida: Path) -> dict:
    r = {}
    err = pred - y
    sq = err ** 2
    r["janelas_afetadas"] = int(afetada.sum())
    jan.loc[afetada, ["id_ciclone", "instante", "vmax", "lon_t", "lat_t"]].assign(previsto_cnn_lstm_kan=pred[afetada]).to_csv(
        saida / "tabelas" / "janelas_afetadas_longitude_trocada.csv", index=False)      # lista para o caso de se decidir descartá-las
    r["pct_sq_afetadas"] = float(100 * sq[afetada].sum() / sq.sum())
    r["rmse_todas"] = float(np.sqrt(sq.mean()))
    r["rmse_sem_afetadas"] = float(np.sqrt(sq[~afetada].mean()))
    # --- composição da faixa
    faixa = (y >= LIMITE_FAIXA_REAL) & (pred <= LIMITE_FAIXA_PREV)
    f = jan.loc[faixa, ["id_ciclone", "instante", "vmax", "lon_t", "lat_t"]].assign(pred=pred[faixa], afetada=afetada[faixa])
    comp = f.groupby("id_ciclone").agg(n=("vmax", "size"), inicio=("instante", "min"), fim=("instante", "max"), vmax_min=("vmax", "min"),
                                       vmax_max=("vmax", "max"), pred_media=("pred", "mean"), lon_media=("lon_t", "mean"),
                                       afetadas=("afetada", "sum")).sort_values("n", ascending=False).reset_index()
    comp.to_csv(saida / "tabelas" / "faixa_composicao_por_ciclone.csv", index=False)
    r["faixa_n"] = int(faixa.sum())
    r["faixa_n_afetadas"] = int(afetada[faixa].sum())
    r["faixa_ciclones"] = int(comp.shape[0])
    # --- faixa conforme o limite de Vmax real (quanto mais intensa a janela, mais a faixa é só o 201407E)
    linhas = []
    for lim in (60, 80, 100, 120):
        m = (y >= lim) & (pred <= LIMITE_FAIXA_PREV)
        linhas.append(dict(vmax_real_min=lim, previsto_max=LIMITE_FAIXA_PREV, n=int(m.sum()), n_201407E=int((m & (jan["id_ciclone"] == CICLONE_LON_TROCADA[0]).to_numpy()).sum()),
                           n_afetadas=int((m & afetada).sum()), ciclones=int(jan.loc[m, "id_ciclone"].nunique())))
    pd.DataFrame(linhas).to_csv(saida / "tabelas" / "faixa_por_limite.csv", index=False)
    r["faixa_por_limite"] = linhas
    # --- erro grande (|erro| ≥ 40) por ciclone e contribuição ao erro quadrático
    grande = np.abs(err) >= 40.0
    r["erro_grande_n"] = int(grande.sum())
    r["erro_grande_afetadas"] = int((grande & afetada).sum())
    por_c = jan.assign(sq=sq, abs_e=np.abs(err), grande=grande, pred=pred, afetada=afetada).groupby("id_ciclone").agg(
        n_janelas=("sq", "size"), rmse=("sq", lambda s: float(np.sqrt(s.mean()))), sq_total=("sq", "sum"), vmax_medio=("vmax", "mean"),
        erro_grande=("grande", "sum"), afetadas=("afetada", "sum"), bacia=("bacia", "first"))
    por_c["pct_sq_total"] = 100 * por_c["sq_total"] / sq.sum()
    top = por_c.sort_values("sq_total", ascending=False).head(15).reset_index()
    top.to_csv(saida / "tabelas" / "erro_por_ciclone_top15.csv", index=False)
    r["top_ciclones_pct_sq"] = {row["id_ciclone"]: float(row["pct_sq_total"]) for _, row in top.head(8).iterrows()}
    cg = jan.loc[grande].groupby("id_ciclone").size().sort_values(ascending=False)
    r["erro_grande_por_ciclone"] = {k: int(v) for k, v in cg.head(10).items()}
    r["erro_grande_ciclones"] = int(cg.shape[0])
    # --- a linha de data em si não é o problema: outros ciclones a ≤ 5° dela
    perto = jan["dist_linha_data"].to_numpy() <= 5.0
    eh = (jan["id_ciclone"] == CICLONE_LON_TROCADA[0]).to_numpy()
    r["perto_5graus"] = dict(
        janelas=int(perto.sum()), ciclones=int(jan.loc[perto, "id_ciclone"].nunique()),
        rmse_outros_ciclones=float(np.sqrt(sq[perto & ~eh].mean())), n_outros=int((perto & ~eh).sum()),
        faixa_outros=int((faixa & perto & ~eh).sum()), erro_grande_outros=int((grande & perto & ~eh).sum()),
        rmse_201407E_afetadas=float(np.sqrt(sq[afetada].mean())), rmse_resto_do_conjunto=float(np.sqrt(sq[~perto].mean())))
    # --- triagem de trilhas refletidas + figuras
    cand = detectar_reflexao(serie)
    cand.to_csv(saida / "tabelas" / "candidatos_reflexao_linha_data.csv", index=False)
    r["candidatos_reflexao"] = cand.assign(instante_topo=cand["instante_topo"].astype(str)).to_dict("records")
    inst = ["2014-08-06 18:00", "2014-08-07 06:00", "2014-08-07 09:00", "2014-08-07 12:00", "2014-08-07 15:00", "2014-08-07 18:00",
            "2014-08-07 21:00", "2014-08-08 00:00", "2014-08-08 12:00", "2014-08-09 00:00", "2014-08-09 12:00", "2014-08-10 15:00"]
    figura_quadros(serie, arqs, CICLONE_LON_TROCADA[0], inst, CICLONE_LON_TROCADA[1], saida / "figuras" / "fig_201407E_quadros_longitude_trocada.png")
    if len(cand):
        figura_candidatos(serie, arqs, cand, saida / "figuras" / "fig_candidatos_reflexao.png")
    trilha = figura_trilha(jan, pred, CICLONE_LON_TROCADA[0], CICLONE_LON_TROCADA[1], saida / "figuras" / "fig_201407E_trilha_e_erro.png")
    trilha.assign(instante=trilha["instante"].astype(str)).to_csv(saida / "tabelas" / "trilha_201407E.csv", index=False)
    figura_faixa(y, pred, afetada, saida / "figuras" / "fig_faixa_antes_depois.png")
    # --- antes × depois da troca, no PRÓPRIO ciclone e só onde Vmax real ≥ 100 (ali "previsto ≈ 25" nunca seria acerto)
    m100 = eh & (y >= 100)
    antes, depois = m100 & ~afetada, m100 & afetada
    r["antes_depois_vmax100"] = dict(
        n_antes=int(antes.sum()), mae_antes=float(np.abs(err[antes]).mean()), previsto_medio_antes=float(pred[antes].mean()),
        n_depois=int(depois.sum()), mae_depois=float(np.abs(err[depois]).mean()), previsto_medio_depois=float(pred[depois].mean()),
        depois_previsto_ate_35=int((depois & (pred <= LIMITE_FAIXA_PREV)).sum()))
    return r


# ----------------------------------------------------------------------------- 8. demais análises
def sobreajuste(execs: list, epocas_plano: int, so_vmax: set) -> pd.DataFrame:
    """Curvas gravadas: RMSE de Vmax na validação interna e perda de treino por época. O *early stopping* (paciência 8) encerra algumas
    execuções antes de `epocas_plano`; as curvas são completadas com NaN e as médias por época usam só as execuções ainda em curso.
    A perda de treino é medida com dropout e rotação e, nas variantes multitarefa, inclui a tarefa de ΔVmax: ali serve só para ver a
    TENDÊNCIA. Nas variantes `so_vmax` a perda é o MSE de Vmax padronizado, então RMSE_treino ≈ √perda × dp(Vmax do treino) (medido com
    dropout e rotação, portanto um limite SUPERIOR do erro de treino em modo de avaliação) e dá para comparar com a validação."""
    linhas = []
    for v in sorted({e["variante"] for e in execs}):
        es = [e for e in execs if e["variante"] == v]
        val = np.full((len(es), epocas_plano), np.nan)
        perda = np.full((len(es), epocas_plano), np.nan)
        for i, e in enumerate(es):
            h = e["meta"]["historico"]
            val[i, :len(h)] = [x["rmse_val_vmax"] for x in h]
            perda[i, :len(h)] = [x["perda_treino"] for x in h]
        n_ep = np.array([len(e["meta"]["historico"]) for e in es])
        melhor = np.array([e["meta"]["melhor_epoca"] + 1 for e in es])          # 1 = primeira época
        pararam = n_ep < epocas_plano
        ultima = lambda M: np.array([M[i, n_ep[i] - 1] for i in range(len(es))])
        rm_treino = rm_val = gap = np.nan
        if v in so_vmax:
            rm_treino = float(np.mean([np.sqrt(e["meta"]["historico"][-1]["perda_treino"]) * e["meta"]["norm"]["vmax"][1] for e in es]))
            rm_val = float(ultima(val).mean())
            gap = rm_val - rm_treino
        linhas.append(dict(variante=v, execucoes=len(es), pararam_antes=int(pararam.sum()),
                           rmse_treino_aprox_ultima=rm_treino, rmse_val_ultima_so_vmax=rm_val, gap_val_menos_treino=gap,
                           melhor_epoca_media=float(melhor.mean()),
                           melhor_epoca_media_das_que_pararam=float(melhor[pararam].mean()) if pararam.any() else np.nan,
                           val_rmse_epoca_10=float(np.nanmean(val[:, 9])), val_rmse_epoca_15=float(np.nanmean(val[:, 14])),
                           val_rmse_epoca_20=float(np.nanmean(val[:, 19])), val_rmse_ultima=float(ultima(val).mean()),
                           val_rmse_no_melhor=float(np.nanmin(val, axis=1).mean()),
                           perda_treino_epoca_5=float(np.nanmean(perda[:, 4])), perda_treino_ultima=float(ultima(perda).mean())))
    return pd.DataFrame(linhas)


def erro_por_intensidade(P: dict, y: np.ndarray, variantes: list) -> pd.DataFrame:
    linhas = []
    for v in variantes:
        pred = np.nanmean(P[v]["vmax"], 0)
        for lo, hi, nome in BINS_INTENSIDADE:
            m = (y >= lo) & (y < hi)
            e = pred[m] - y[m]
            linhas.append(dict(variante=v, faixa=nome, n=int(m.sum()), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), vies=float(np.mean(e))))
    return pd.DataFrame(linhas)


def resultados_2017(P: dict, y: np.ndarray, jan: pd.DataFrame, variantes: list) -> pd.DataFrame:
    """Janelas de 2017 (todas, juntando os 5 folds, como numa avaliação única): RMSE/MAE de cada seed, média e dp entre as seeds.
    NÃO é a comparação do TCC: lá o treino era 2003–2014 e as imagens eram avaliadas isoladas; aqui cada janela de 2017 foi prevista
    por um modelo treinado nos outros folds (que incluem outros ciclones de 2017)."""
    m = jan["ano"].to_numpy() == 2017
    linhas = []
    for v in variantes:
        rm, ma = [], []
        for i in range(P[v]["vmax"].shape[0]):
            met = metricas_regressao(y[m], P[v]["vmax"][i][m])
            rm.append(met["rmse"])
            ma.append(met["mae"])
        linhas.append(dict(modelo=v, rmse_2017=float(np.mean(rm)), rmse_dp_seeds=float(np.std(rm, ddof=1)), mae_2017=float(np.mean(ma)),
                           n_janelas_2017=int(m.sum())))
    for nome, (rmse, mae) in TCC_REF.items():
        linhas.append(dict(modelo=nome, rmse_2017=rmse, rmse_dp_seeds=np.nan, mae_2017=mae, n_janelas_2017=4296))
    return pd.DataFrame(linhas)


def media_seeds(M: np.ndarray) -> np.ndarray:
    """Média entre as seeds (eixo 0) ignorando NaN; NaN onde nenhuma seed tem valor (sem avisos de 'mean of empty slice')."""
    cnt = (~np.isnan(M)).sum(0)
    return np.where(cnt > 0, np.nansum(M, 0) / np.maximum(cnt, 1), np.nan)


def resumo_dvmax(dv: np.ndarray, pred: np.ndarray, fold: np.ndarray) -> dict:
    """RMSE e MAE (média dos 5 folds, como a tabela do notebook) e AUC/AP de RI com os 5 folds juntos."""
    ok = ~np.isnan(dv)
    por_fold = [metricas_regressao(dv[fold == f], pred[fold == f]) for f in range(ej.N_FOLDS)]
    real = dv[ok] >= LIMIAR_RI
    return dict(rmse=float(np.mean([m["rmse"] for m in por_fold])), mae=float(np.mean([m["mae"] for m in por_fold])),
                auc=auc_roc(real, pred[ok]), ap=precisao_media(real, pred[ok]))


def correlacoes_latentes(caminho: Path) -> dict:
    if not caminho.exists():
        return {}
    c = pd.read_csv(caminho, index_col=0)
    return {col: dict(variavel=str(c[col].abs().idxmax()), rho=float(c.loc[c[col].abs().idxmax(), col])) for col in c.columns}


# ----------------------------------------------------------------------------- relatório
def escrever_relatorio(R: dict, T: dict, caminho: Path) -> None:
    L = []
    add = L.append
    add("# Análise da execução completa da pipeline (Kaggle)\n")
    add("Gerado por `analisar_resultados_pipeline.py` a partir das predições gravadas (`runs/*.npz`). Números de desempenho: média entre as "
        "execuções (fold × seed), em nós.\n")
    e = R["execucao"]
    add("## 1. Execução\n")
    add(f"- {e['execucoes']} execuções ({', '.join(e['variantes'])}), assinatura `{e['assinatura']}`; plano completo: {'sim' if e['completo'] else 'NÃO'}.")
    if "gpu" in R["log"]:
        add(f"- Ambiente (log): {R['log']['gpu']['nome']} ({R['log']['gpu']['memoria_gb']} GB), torch {R['log'].get('ambiente', {}).get('torch', '?')}.")
    if "tempo_treino" in R["log"]:
        add(f"- Tempo de treino (log): {R['log']['tempo_treino']['horas']} h ({R['log']['tempo_treino']['inicio']} → {R['log']['tempo_treino']['fim']}).")
    add(f"- Janelas reconstruídas: {R['validacao']['janelas']} de {R['validacao']['ciclones']} ciclones; as {R['validacao']['execucoes_conferidas']} "
        "execuções batem com elas em rótulos e folds.")
    add(f"- Conferência com `metricas_por_execucao.csv` do notebook (maior diferença absoluta): "
        + ", ".join(f"{k} {fmt(v, 4)}" for k, v in R["conferencia_notebook"].items() if k != "linhas_comparadas") + ".\n")
    add("## 2. Resultados por variante (média ± dp entre as 25 execuções)\n")
    add(tabela_md(T["resumo"]) + "\n")
    add("## 3. RMSE de Vmax por fold e variação entre folds × entre seeds\n")
    add(tabela_md(T["variacao"]) + "\n")
    add("## 4. Bootstrap por ciclone (réplica do notebook)\n")
    add(f"Maior diferença contra `bootstrap_por_ciclone.csv` do notebook: dif {fmt(R['bootstrap_conferencia']['dif'], 6)}, "
        f"IC {fmt(R['bootstrap_conferencia']['ic'], 6)}, p {fmt(R['bootstrap_conferencia']['p'], 6)}.\n")
    add(tabela_md(T["bootstrap"].drop(columns=["significativo"])) + "\n")
    add("## 5. Faixa horizontal do gráfico predito × real\n")
    f = R["faixa"]
    add(f"- Faixa = Vmax real ≥ {LIMITE_FAIXA_REAL:.0f} e previsto ≤ {LIMITE_FAIXA_PREV:.0f} nós: **{f['faixa_n']} janelas** de {f['faixa_ciclones']} ciclones; "
        f"{f['faixa_n_afetadas']} delas são do {CICLONE_LON_TROCADA[0]} depois da troca de sinal da longitude.")
    add(tabela_md(pd.DataFrame(f["faixa_por_limite"])) + "\n")
    add(f"- Janelas afetadas ({CICLONE_LON_TROCADA[0]} a partir de {CICLONE_LON_TROCADA[1]:%d/%m/%Y %Hh}): **{f['janelas_afetadas']}**, "
        f"{fmt(f['pct_sq_afetadas'], 2)}% do erro quadrático; RMSE {fmt(f['rmse_todas'], 3)} → {fmt(f['rmse_sem_afetadas'], 3)} sem elas (cnn_lstm_kan, média das seeds).")
    p5 = f["perto_5graus"]
    add(f"- Outros ciclones a ≤ 5° da linha de data: {p5['n_outros']} janelas, RMSE {fmt(p5['rmse_outros_ciclones'], 2)} (resto do conjunto: "
        f"{fmt(p5['rmse_resto_do_conjunto'], 2)}); na faixa: {p5['faixa_outros']}; com erro ≥ 40 nós: {p5['erro_grande_outros']}.")
    ad = f["antes_depois_vmax100"]
    add(f"- Mesmo ciclone ({CICLONE_LON_TROCADA[0]}), só janelas com Vmax real ≥ 100 nós: antes da troca n = {ad['n_antes']}, erro absoluto médio "
        f"{fmt(ad['mae_antes'], 1)} nós (previsto médio {fmt(ad['previsto_medio_antes'], 1)}); depois da troca n = {ad['n_depois']}, erro absoluto médio "
        f"{fmt(ad['mae_depois'], 1)} nós (previsto médio {fmt(ad['previsto_medio_depois'], 1)}; {ad['depois_previsto_ate_35']} delas previstas em ≤ 35 nós).")
    add(f"- Erro ≥ 40 nós: {f['erro_grande_n']} janelas de {f['erro_grande_ciclones']} ciclones; as mais frequentes: "
        + ", ".join(f"{k} ({v})" for k, v in f["erro_grande_por_ciclone"].items()) + ".")
    add("- Triagem de trilhas refletidas (aid; confirmar nos quadros): " + (", ".join(
        f"{c['id_ciclone']} ({c['instante_topo'][:10]}, lon {c['lon_topo']:.1f}°)" for c in f["candidatos_reflexao"]) or "nenhuma") + ".\n")
    add("### Top 15 ciclones por erro quadrático (cnn_lstm_kan)\n")
    add(tabela_md(T["top_ciclones"]) + "\n")
    add("## 6. Sensibilidade: sem as janelas afetadas\n")
    add(tabela_md(T["sensibilidade_metricas"]) + "\n")
    add(tabela_md(T["bootstrap_sem_afetadas"].drop(columns=["significativo"])) + "\n")
    add("## 7. ΔVmax e RI: modelo único × ensemble, histórico e empilhamento\n")
    add("RMSE/MAE: média dos 5 folds. AUC e AP de RI: os 5 folds juntos (AUC sem habilidade = 0,5; AP sem habilidade ≈ prevalência 0,067). "
        "`_dp`: desvio entre as 5 seeds (só nas linhas de modelo único). **Ensemble = média das previsões das 5 seeds**: é outro modelo, mais "
        "caro e com menos variância; a comparação justa com o baseline de histórico (um único ajuste linear) é a do modelo único, e a do ensemble "
        "deve ser rotulada como tal.\n")
    add(tabela_md(T["dvmax_tabela"]) + "\n")
    add("### Bootstrap por ciclone com o ENSEMBLE das seeds (referência − outra; negativo = referência erra menos)\n")
    add(tabela_md(T["bootstrap_ensemble"].drop(columns=["significativo"])) + "\n")
    if len(T["ri_bootstrap"]):
        add("### RI: AUC e AP com IC 95% (bootstrap por ciclone, pareado; escores = ΔVmax previsto dos ensembles; diferenças em relação à 1ª linha)\n")
        add(tabela_md(T["ri_bootstrap"]) + "\n")
    add("## 8. Sobreajuste (curvas gravadas)\n")
    add(tabela_md(T["sobreajuste"]) + "\n")
    add("## 9. Erro por faixa de intensidade (média das seeds)\n")
    add(tabela_md(T["intensidade"]) + "\n")
    add("## 10. Janelas de 2017 (referência ao TCC; protocolos diferentes)\n")
    add(tabela_md(T["r2017"]) + "\n")
    add("## 11. KAN (log do notebook)\n")
    add("```json\n" + json.dumps(json_seguro(R["log"].get("kan", {})), ensure_ascii=False, indent=2) + "\n```\n")
    add("Variável latente mais correlacionada (Spearman) com cada grandeza: " + json.dumps(json_seguro(R["latentes"]), ensure_ascii=False) + "\n")
    caminho.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- principal
def main() -> None:
    ap = argparse.ArgumentParser(description="Análise da execução completa da pipeline (zip do Kaggle).")
    ap.add_argument("--entrada", default=str(RAIZ / "resultados_pipeline" / "saida_pipeline"), help="pasta saida_pipeline/ extraída do zip")
    ap.add_argument("--notebook", default=None, help="notebook executado (padrão: ../__notebook__.ipynb da entrada)")
    ap.add_argument("--dados", default=str(RAIZ / "data"), help="pasta com os 3 TCIR-*.h5 completos")
    ap.add_argument("--cache-ir", default=str(RAIZ / "resultados_janelas" / "frac_invalido_ir.csv"))
    ap.add_argument("--saida", default=str(RAIZ / "resultados_analise_pipeline"))
    ap.add_argument("--reps", type=int, default=2000, help="réplicas do bootstrap por ciclone (o notebook usou 2000)")
    ap.add_argument("--reps-ri", type=int, default=2000, help="réplicas do bootstrap de AUC/AP (0 = pula)")
    args = ap.parse_args()

    ent, saida = Path(args.entrada), Path(args.saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ent / "config.json").read_text(encoding="utf-8"))
    resumo_nb = json.loads((ent / "resumo_final.json").read_text(encoding="utf-8"))
    seeds, variantes, ref = list(cfg["seeds"]), list(cfg["variantes"]), cfg["variante_referencia"]
    R, T = {}, {}

    # 1) execuções e log
    log("lendo as execuções (.npz)")
    execs = carregar_execucoes(ent / "runs")
    esperado = {(v, f, s) for v in variantes for f in range(cfg["n_folds"]) for s in seeds}
    achado = {(e["variante"], e["fold"], e["seed"]) for e in execs}
    assinaturas = sorted({e["assinatura"] for e in execs})
    R["execucao"] = dict(execucoes=len(execs), variantes=variantes, assinatura=",".join(assinaturas), completo=(esperado == achado and len(execs) == len(esperado)),
                         faltando=sorted(esperado - achado), modo=cfg["modo"], assinatura_notebook=resumo_nb.get("assinatura"))
    nb_path = Path(args.notebook) if args.notebook else ent.parent / "__notebook__.ipynb"
    R["log"] = analisar_log(ler_log_notebook(nb_path)) if nb_path.exists() else {}
    runs_log = R["log"].pop("runs_log", pd.DataFrame())
    log(f"{len(execs)} execuções | assinaturas {assinaturas} | log do notebook: {'ok' if R['log'] else 'ausente'} ({len(runs_log)} linhas de treino)")

    # 2) janelas e validação
    log("reconstruindo as janelas (estatisticas_janelas.py)")
    jan, serie, fold, arqs = reconstruir_janelas(Path(args.dados), Path(args.cache_ir))
    R["validacao"] = validar_mapeamento(execs, jan, fold)
    log(f"mapeamento validado: {R['validacao']}")
    N = len(jan)
    y, dv = jan["vmax"].to_numpy(float), jan["dvmax"].to_numpy(float)
    P = matrizes_previsao(execs, variantes, seeds, N)
    cobertura = {v: bool((~np.isnan(P[v]["vmax"])).all()) for v in variantes}
    assert all(cobertura.values()), f"alguma (variante, seed) não cobre todas as janelas: {cobertura}"
    pred_ref = np.nanmean(P[ref]["vmax"], 0)
    afetada = ((jan["id_ciclone"] == CICLONE_LON_TROCADA[0]) & (jan["instante"] >= CICLONE_LON_TROCADA[1])).to_numpy()

    # 3) métricas por execução
    df = metricas_por_execucao(execs)
    df.to_csv(saida / "tabelas" / "metricas_por_execucao_recalculadas.csv", index=False)
    R["conferencia_notebook"] = conferir_com_notebook(df, ent / "tabelas" / "metricas_por_execucao.csv")
    if len(runs_log):
        c = df.merge(runs_log, on=["variante", "fold", "seed"], suffixes=("", "_log"))
        R["conferencia_log"] = dict(linhas=int(len(c)), max_dif_rmse_vmax=float((c["rmse_vmax"] - c["rmse_vmax_log"]).abs().max()),
                                    max_dif_mae_vmax=float((c["mae_vmax"] - c["mae_vmax_log"]).abs().max()),
                                    soma_segundos_treino_log=int(runs_log["segundos"].sum()))
    resumo = df.groupby("variante").agg(execucoes=("seed", "size"), rmse_vmax=("rmse_vmax", "mean"), rmse_vmax_dp=("rmse_vmax", "std"), mae_vmax=("mae_vmax", "mean"),
                                        vies_vmax=("vies_vmax", "mean"), rmse_dvmax=("rmse_dvmax", "mean"), rmse_dvmax_dp=("rmse_dvmax", "std"),
                                        mae_dvmax=("mae_dvmax", "mean"), auc_ri=("auc_ri", "mean"), ap_ri=("ap_ri", "mean"), csi_ri_aj=("csi_ri_aj", "mean"),
                                        melhor_epoca=("melhor_epoca", "mean"), minutos_por_execucao=("segundos", lambda s: s.mean() / 60)).reset_index()
    resumo = resumo.set_index("variante").loc[variantes].reset_index()
    ens = []
    for v in variantes:
        m = metricas_regressao(y, np.nanmean(P[v]["vmax"], 0))
        ens.append(m["rmse"])
    resumo["rmse_vmax_ensemble_seeds"] = ens
    resumo.to_csv(saida / "tabelas" / "resumo_por_variante.csv", index=False)
    T["resumo"] = resumo
    T["variacao"] = decompor_variacao(df)
    T["variacao"].to_csv(saida / "tabelas" / "variacao_entre_folds_e_seeds.csv", index=False)

    # 4) bootstrap por ciclone: réplica do notebook + sensibilidade
    erros = {v: {"vmax": erros_medios(P[v]["vmax"], y), "dvmax": erros_medios(P[v]["dvmax"], dv)} for v in variantes}
    prev_hist, _ = baseline_historico(jan, serie, fold)
    nan = np.full(N, np.nan)
    erros["baseline_historico"] = {"vmax": (nan, nan), "dvmax": ((np.abs(prev_hist - dv)), (prev_hist - dv) ** 2)}
    outras = [v for v in variantes if v != ref] + ["baseline_historico"]
    ids = jan["id_ciclone"].to_numpy()
    log(f"bootstrap por ciclone ({args.reps} réplicas) — réplica do notebook")
    bt = bootstrap_por_ciclone(erros, ref, outras, ids, args.reps, SEED_ESTRUTURAL)
    bt.to_csv(saida / "tabelas" / "bootstrap_replicado.csv", index=False)
    T["bootstrap"] = bt
    nb_bt = pd.read_csv(ent / "tabelas" / "bootstrap_por_ciclone.csv")
    mm = bt.merge(nb_bt, on=["comparacao", "tarefa", "metrica"], suffixes=("", "_nb"))
    R["bootstrap_conferencia"] = dict(
        linhas=int(len(mm)), dif=float((mm["dif"] - mm["dif_nb"]).abs().max()),
        ic=float(max((mm["ic95_inf"] - mm["ic95_inf_nb"]).abs().max(), (mm["ic95_sup"] - mm["ic95_sup_nb"]).abs().max())),
        p=float((mm["p_holm"] - mm["p_holm_nb"]).abs().max()), veredito_igual=bool((mm["veredito"] == mm["veredito_nb"]).all()))
    # baseline: confere por fold com o notebook
    base_nb = pd.read_csv(ent / "tabelas" / "baselines_dvmax_por_fold.csv")
    base_nb = base_nb[base_nb["variante"] == "baseline_historico"].set_index("fold")["rmse_dvmax"]
    R["baseline_conferencia_rmse_por_fold"] = float(max(abs(metricas_regressao(dv[fold == f], prev_hist[fold == f])["rmse"] - base_nb[f]) for f in range(ej.N_FOLDS)))

    # 5) a faixa horizontal e a longitude trocada
    log("investigando a faixa horizontal (figuras: lê alguns quadros dos .h5)")
    R["faixa"] = analisar_faixa(jan, serie, arqs, y, pred_ref, afetada, saida)
    top = pd.read_csv(saida / "tabelas" / "erro_por_ciclone_top15.csv")
    T["top_ciclones"] = top[["id_ciclone", "bacia", "n_janelas", "rmse", "pct_sq_total", "vmax_medio", "erro_grande", "afetadas"]]

    # 6) sensibilidade: sem as janelas afetadas
    mascara = ~afetada
    log("bootstrap por ciclone sem as janelas afetadas")
    bts = bootstrap_por_ciclone(erros, ref, outras, ids, args.reps, SEED_ESTRUTURAL, mascara=mascara)
    bts.to_csv(saida / "tabelas" / "bootstrap_sem_afetadas.csv", index=False)
    T["bootstrap_sem_afetadas"] = bts
    linhas = []
    for v in variantes:
        com, sem = [], []
        for e in execs:
            if e["variante"] != v:
                continue
            m = mascara[e["idx_teste"]]
            com.append(metricas_regressao(e["y_vmax"], e["pred_vmax"])["rmse"])
            sem.append(metricas_regressao(e["y_vmax"][m], e["pred_vmax"][m])["rmse"])
        d_com = [metricas_regressao(e["y_dvmax"], e["pred_dvmax"])["rmse"] for e in execs if e["variante"] == v]
        d_sem = [metricas_regressao(e["y_dvmax"][mascara[e["idx_teste"]]], e["pred_dvmax"][mascara[e["idx_teste"]]])["rmse"] for e in execs if e["variante"] == v]
        linhas.append(dict(variante=v, rmse_vmax=np.mean(com), rmse_vmax_sem_afetadas=np.mean(sem), dif=np.mean(sem) - np.mean(com),
                           rmse_dvmax=np.nanmean(d_com) if not np.isnan(d_com).all() else np.nan,
                           rmse_dvmax_sem_afetadas=np.nanmean(d_sem) if not np.isnan(d_sem).all() else np.nan))
    T["sensibilidade_metricas"] = pd.DataFrame(linhas)
    T["sensibilidade_metricas"].to_csv(saida / "tabelas" / "sensibilidade_metricas.csv", index=False)

    # 7) ensemble das seeds, ΔVmax × histórico, empilhamento e RI
    # Um MODELO ÚNICO (uma seed) e um ENSEMBLE (média das previsões das 5 seeds) são coisas diferentes: o ensemble reduz a variância
    # do treino. As tabelas do notebook são de modelo único; aqui o ensemble entra como comparação à parte e rotulada.
    ens = {v: {t: media_seeds(P[v][t]) for t in ("vmax", "dvmax")} for v in variantes}
    erros_ens = {v: {"vmax": (np.abs(ens[v]["vmax"] - y), (ens[v]["vmax"] - y) ** 2),
                     "dvmax": (np.abs(ens[v]["dvmax"] - dv), (ens[v]["dvmax"] - dv) ** 2)} for v in variantes}
    erros_ens["baseline_historico"] = erros["baseline_historico"]
    log("bootstrap por ciclone com o ENSEMBLE das seeds")
    bte = bootstrap_por_ciclone(erros_ens, ref, outras, ids, args.reps, SEED_ESTRUTURAL)
    bte.to_csv(saida / "tabelas" / "bootstrap_ensemble.csv", index=False)
    T["bootstrap_ensemble"] = bte

    prev_hist2, prev_emp_ens = baseline_historico(jan, serie, fold, p_rede=ens[ref]["dvmax"])
    assert np.allclose(prev_hist2, prev_hist, equal_nan=True)
    emp_seed = [baseline_historico(jan, serie, fold, p_rede=P[ref]["dvmax"][i])[1] for i in range(len(seeds))]

    def agrega(nome: str, lista: list) -> dict:
        out = dict(modelo=nome)
        for k in ("rmse", "mae", "auc", "ap"):
            x = np.array([d[k] for d in lista], dtype=float)
            out[k] = float(x.mean())
            out[k + "_dp"] = float(x.std(ddof=1)) if len(x) > 1 else np.nan
        return out

    linhas = []
    for v in variantes:
        if np.isnan(P[v]["dvmax"]).all():
            continue                                                         # variante só-Vmax: sem ΔVmax
        linhas.append(agrega(f"{v} — modelo único (média das {len(seeds)} seeds)", [resumo_dvmax(dv, P[v]["dvmax"][i], fold) for i in range(len(seeds))]))
        linhas.append(agrega(f"{v} — ensemble das {len(seeds)} seeds", [resumo_dvmax(dv, ens[v]["dvmax"], fold)]))
    linhas.append(agrega("baseline_historico (sem imagens)", [resumo_dvmax(dv, prev_hist, fold)]))
    linhas.append(agrega(f"histórico + {ref} — modelo único (média das seeds)", [resumo_dvmax(dv, p, fold) for p in emp_seed]))
    linhas.append(agrega(f"histórico + {ref} — ensemble", [resumo_dvmax(dv, prev_emp_ens, fold)]))
    T["dvmax_tabela"] = pd.DataFrame(linhas)
    T["dvmax_tabela"].to_csv(saida / "tabelas" / "dvmax_modelo_unico_ensemble_empilhamento.csv", index=False)
    if args.reps_ri > 0:
        log(f"bootstrap pareado de AUC/AP de RI por ciclone ({args.reps_ri} réplicas; leva alguns minutos)")
        T["ri_bootstrap"] = bootstrap_ri(jan, {f"{ref} (ensemble)": ens[ref]["dvmax"], "cnn_lstm_mlp (ensemble)": ens["cnn_lstm_mlp"]["dvmax"],
                                              "cnn_kan_1frame (ensemble)": ens["cnn_kan_1frame"]["dvmax"], "baseline_historico": prev_hist,
                                              "histórico + rede (ensemble)": prev_emp_ens}, args.reps_ri, SEED_ESTRUTURAL + 1)
    else:
        T["ri_bootstrap"] = pd.DataFrame()
    T["ri_bootstrap"].to_csv(saida / "tabelas" / "ri_auc_ap_bootstrap.csv", index=False)

    # 8) sobreajuste, intensidade, 2017, KAN
    so_vmax = {v for v in variantes if np.isnan(P[v]["dvmax"]).all()}
    T["sobreajuste"] = sobreajuste(execs, int(cfg["epocas_max"]), so_vmax)
    T["sobreajuste"].to_csv(saida / "tabelas" / "sobreajuste_por_variante.csv", index=False)
    T["intensidade"] = erro_por_intensidade(P, y, variantes)
    T["intensidade"].to_csv(saida / "tabelas" / "erro_por_intensidade.csv", index=False)
    T["r2017"] = resultados_2017(P, y, jan, variantes)
    T["r2017"].to_csv(saida / "tabelas" / "resultados_2017.csv", index=False)
    R["latentes"] = correlacoes_latentes(ent / "tabelas" / f"kan_latentes_correlacoes_{ref}.csv")

    # 9) saídas
    R["comparacao_com_resumo_do_notebook"] = {v: dict(rmse_vmax=float(resumo.set_index("variante").loc[v, "rmse_vmax"]),
                                                      nb=resumo_nb["resultados"][v]["rmse_vmax"]) for v in variantes}
    R["tabelas"] = {k: v.to_dict("records") for k, v in T.items() if isinstance(v, pd.DataFrame)}
    (saida / "resumo_analise.json").write_text(json.dumps(json_seguro(R), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    escrever_relatorio(R, T, saida / "relatorio_analise.md")
    log(f"pronto: {saida}/relatorio_analise.md e resumo_analise.json")


if __name__ == "__main__":
    main()
