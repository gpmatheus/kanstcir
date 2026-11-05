#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisar_rodada_rapida.py — confere e resume UMA rodada do notebook rápido baixada do Kaggle (versão SEM folds)
==============================================================================================================

Contexto: cada rodada do notebook rápido (`pipeline_tc_kaggle_rapido.ipynb`) devolve um `.zip` com a pasta `<NOME_EXPERIMENTO>_seed<seed>/`
(`config.json`, `resumo_final.json`, `RELATORIO.md`, `runs/`, `tabelas/`, `figuras/`; em modo teste o nome ganha o sufixo `_TESTE`). Desde 2026-10-23
o notebook rápido não usa folds: há UMA divisão (treino / validação / teste; por padrão por ano, como no TCC) e um treino por (variante, seed), com
arquivos `runs/<variante>__s<seed>__<assinatura>.npz`. Este script não treina nada: confere que a rodada saiu inteira e coerente, recalcula as
métricas a partir das predições e escreve um relatório curto. A pasta baixada é dado não confiável: só se leem JSON, CSV e `.npz` (com
`allow_pickle=False`); os `.pt` (pesos, em formato pickle) nunca são abertos.

Rodadas ANTIGAS (com 5 folds; arquivos `<variante>__f<fold>__s<seed>__...npz`) NÃO são lidas por este script: use a versão guardada em
`versoes_anteriores/analisar_rodada_rapida_com_folds.py` (ela reproduz a análise da rodada completa de 2026-10-22/23).

O QUE FAZ
 1. identifica a rodada (modo, dispositivo, assinatura, plano, dados, divisão) e avisa em destaque se é MODO TESTE (números sem valor);
 2. confere a integridade (seção 3 do relatório): plano completo, mesma assinatura em todos os arquivos, partição das janelas, mesmas janelas de
    teste em todas as execuções, métricas recalculadas das predições iguais às do notebook, melhor época coerente com o histórico, colunas novas
    das curvas preenchidas, predições finitas, duplicatas e arquivos citados no RELATORIO.md;
 3. resume o desempenho (e, com a divisão por ano, compara com o TCC: treino 2003–2014, teste 2017) e as curvas de treino (treino em modo de
    avaliação × validação, por época, em Vmax e ΔVmax; o ΔVmax no melhor ponto);
 4. confere o AP (precisão média) com empates (seção 6): a `precisao_media` do notebook ordena os escores iguais pela ordem das janelas, então um
    escore constante (baselines «zero» e «média») não dá AP = prevalência. A versão deste script soma sobre os limiares distintos (a definição de
    `sklearn.metrics.average_precision_score`) e é conferida contra a definição direta;
 5. numa rodada completa: erro por faixa de intensidade, janelas de erro grande (quantas, que parcela do erro quadrático, de quais ciclones) e as
    variáveis latentes da KAN mais correlacionadas com cada grandeza;
 6. RI (intensificação rápida: ΔVmax real ≥ 30 nós em 24 h; seção 4): quantos eventos há no teste, o que a rede previu de ΔVmax e o quadro de
    alarmes e acertos (TP, FP, FN) com o limiar fixo e com o ajustado na validação; POD, FAR e CSI são recalculados das predições e conferidos
    contra a tabela do notebook (o limiar ajustado também é recalculado, da previsão da validação gravada no `.npz`).

USO
    python3 -I scripts/analise/analisar_rodada_rapida.py --entrada <pasta>/<nome>_seed<k> [--saida resultados_analise_rapido]

SAÍDAS (em `resultados_analise_rapido/<nome da pasta da rodada>/`)
    relatorio.md        o relatório (também impresso no terminal)
    resumo.json         números principais e o resultado de cada verificação
    verificacoes.csv    uma linha por verificação
"""
import argparse
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

DUPLICATAS_TCIR = dict(registros_removidos=72, ids_unificados=2, pares_identicos_bit_a_bit=70)   # `verificar_duplicatas.py`, TCIR inteiro
IMAGENS_TCIR = 75079                                        # imagens do TCIR inteiro (só então vale comparar as duplicatas com o esperado)
TOL_METRICAS = 1e-3                                         # nós: as predições são float32 e as métricas do notebook saem delas
TOL_RI = 1e-5                                               # POD, FAR e CSI são razões de contagens; o CSV do notebook as grava com 6 casas
N_CASOS_AUTOTESTE = 300                                     # casos aleatórios do autoteste do AP
COLUNAS_CURVAS = ("perda_val", "perda_treino_aval", "rmse_treino_vmax")        # colunas novas do histórico, de toda variante
COLUNAS_CURVAS_DVMAX = ("rmse_val_dvmax", "rmse_treino_dvmax")                 # colunas novas só de quem tem cabeça de ΔVmax
EPOCAS_CURVA_TREINO = (1, 5, 10, 15, 20, 25)                # épocas em que se lê a folga treino × validação (as que passam do que a rodada rodou saem)
BINS_INTENSIDADE = [(10, 34, "DT (< 34)"), (34, 64, "TS (34–63)"), (64, 83, "Cat 1 (64–82)"), (83, 96, "Cat 2 (83–95)"),
                    (96, 113, "Cat 3 (96–112)"), (113, 137, "Cat 4 (113–136)"), (137, 1000, "Cat 5 (≥ 137)")]       # as de `analisar_resultados_pipeline.py`
DESCRICAO_VARIANTE = {"cnn_lstm_kan": "a proposta: CNN + LSTM + cabeças KAN, multitarefa (Vmax e ΔVmax)",
                      "cnn_lstm_mlp": "controle da KAN: igual à proposta, com cabeça MLP no lugar da KAN",
                      "cnn_kan_1frame": "controle da LSTM: sem sequência (só o quadro t), cabeças KAN",
                      "cnn_lstm_kan_vmax": "controle da multitarefa: igual à proposta, treinada só com Vmax"}
REF_TCC = {"CNN-TC (TCC)": (10.06, 7.53), "MobileNetV2+SDTA (TCC)": (9.14, 7.00)}       # (RMSE, MAE) em nós no teste de 2017 do TCC (N = 4296 imagens)
_NOME_NPZ = re.compile(r"^(?P<variante>.+)__s(?P<seed>\d+)__(?P<assinatura>[0-9a-f]+)\.npz$")


# ----------------------------------------------------------------------------- 1. leitura
def ler_execucoes(pasta: Path) -> list:
    """Lê os `runs/*.npz` (arrays + `meta` em JSON), sem pickle."""
    execs = []
    for arq in sorted(pasta.glob("*.npz")):
        m = _NOME_NPZ.match(arq.name)
        if m is None:
            raise SystemExit(f"nome de execução inesperado: {arq.name} (rodada com folds? use versoes_anteriores/analisar_rodada_rapida_com_folds.py)")
        with np.load(arq, allow_pickle=False) as z:
            e = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        e.update(arquivo=arq.name, variante=m["variante"], seed=int(m["seed"]), assinatura=m["assinatura"], meta=meta)
        execs.append(e)
    return execs


def dados_do_relatorio(txt: str) -> dict:
    """Dispositivo e hora de geração, do cabeçalho do RELATORIO.md (o `config.json` não traz o nome da GPU)."""
    disp, quando = re.search(r"dispositivo:\*\*\s*(.+?)\s*·", txt), re.search(r"Gerado em:\*\*\s*(.+?)\s*·", txt)
    return dict(dispositivo=disp.group(1) if disp else "?", gerado_em=quando.group(1) if quando else "?")


def arquivos_do_relatorio(txt: str) -> list:
    """Arquivos de `figuras/` e `tabelas/` que a seção «Arquivos desta rodada» do RELATORIO.md diz ter gerado."""
    achados, pasta = [], None
    for linha in txt.splitlines():
        if linha.startswith("**figuras/**"):
            pasta = "figuras"
        elif linha.startswith("**tabelas/**"):
            pasta = "tabelas"
        elif linha.startswith("**runs/**"):
            pasta = None
        m = re.match(r"- `([^`]+)`", linha)
        if m and pasta:
            achados.append((pasta, m.group(1)))
    return achados


def duracao_do_notebook(pasta: Path):
    """Minutos entre a gravação do `config.json` (início do notebook) e a do `RELATORIO.md` (fim), pelos carimbos dos arquivos."""
    a, b = (pasta / "config.json").stat().st_mtime, (pasta / "RELATORIO.md").stat().st_mtime
    return (b - a) / 60 if 0 < b - a < 86400 else None


def linha_do_tempo(pasta: Path):
    """Minutos de cada fase da rodada, pelos carimbos dos arquivos: do `config.json` à 1ª figura (dados, duplicatas, janelas e cache), dali ao último treino
    (`runs/*.npz`) e do último treino ao `RELATORIO.md` (tabelas, estatística, Grad-CAM e figuras). None se faltar arquivo ou a ordem não fizer sentido."""
    try:
        t0, t1 = (pasta / "config.json").stat().st_mtime, (pasta / "figuras" / "00_distribuicoes.png").stat().st_mtime
        t3, t4 = max(p.stat().st_mtime for p in (pasta / "runs").glob("*.npz")), (pasta / "RELATORIO.md").stat().st_mtime
    except (OSError, ValueError):
        return None
    return dict(preparo=(t1 - t0) / 60, treino=(t3 - t1) / 60, depois=(t4 - t3) / 60) if t0 < t1 < t3 < t4 else None


def caminho_relativo(p: Path, raiz: Path) -> str:
    """Caminho relativo à raiz do projeto (relatório portável); o próprio caminho se estiver fora dela."""
    try:
        return str(p.resolve().relative_to(raiz))
    except ValueError:
        return str(p)


# ----------------------------------------------------------------------------- 2. métricas (réplica das do notebook)
def erros(y: np.ndarray, p: np.ndarray) -> dict:
    """RMSE, MAE e viés (previsto − real) nas janelas em que os dois existem."""
    ok = ~(np.isnan(y) | np.isnan(p))
    if not ok.any():
        return dict(n=0, rmse=np.nan, mae=np.nan, vies=np.nan)
    d = p[ok].astype(np.float64) - y[ok]
    return dict(n=int(ok.sum()), rmse=float(np.sqrt(np.mean(d ** 2))), mae=float(np.mean(np.abs(d))), vies=float(np.mean(d)))


def auc_roc(real: np.ndarray, escore: np.ndarray) -> float:
    """Área sob a curva ROC (empate vale 1/2); cópia da função do notebook."""
    n1, n0 = int(real.sum()), int((~real).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = sps.rankdata(escore)
    return float((r[real].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap_do_notebook(real: np.ndarray, escore: np.ndarray) -> float:
    """Cópia literal de `precisao_media` (notebooks e `analisar_resultados_pipeline.py`): escores iguais saem na ordem das janelas."""
    if real.sum() == 0:
        return np.nan
    r = real[np.argsort(-escore, kind="stable")]
    acertos = np.cumsum(r)
    return float((acertos / np.arange(1, len(r) + 1))[r].sum() / r.sum())


def ap_com_empates(real: np.ndarray, escore: np.ndarray) -> float:
    """Precisão média sobre os limiares DISTINTOS: AP = Σ (R_n − R_{n−1}) · P_n, com os escores iguais entrando juntos.
    Para um escore constante dá a prevalência; sem empates coincide com `precisao_media` do notebook."""
    if real.sum() == 0:
        return np.nan
    o = np.argsort(-escore, kind="stable")
    s, r = escore[o], real[o]
    fim = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]       # último índice de cada grupo de escores iguais
    tp = np.cumsum(r)[fim]
    return float(np.sum(np.diff(np.r_[0.0, tp / real.sum()]) * (tp / (fim + 1))))


def _ap_forca_bruta(real: np.ndarray, escore: np.ndarray) -> float:
    """Definição direta, sem vetorizar: para cada limiar distinto (do maior ao menor), precisão e revocação de «escore ≥ limiar»."""
    ap, rev_ant = 0.0, 0.0
    for t in sorted(set(escore.tolist()), reverse=True):
        sel = escore >= t
        tp = int((real & sel).sum())
        rev = tp / int(real.sum())
        ap += (rev - rev_ant) * tp / int(sel.sum())
        rev_ant = rev
    return ap


def autoteste_ap(n_casos: int = N_CASOS_AUTOTESTE, seed: int = 0) -> tuple:
    """Maior diferença para a força bruta: de `ap_com_empates` em escores com muitos empates, e de `ap_do_notebook` e `ap_com_empates`
    em escores sem empates (onde as duas versões têm de coincidir)."""
    rng = np.random.default_rng(seed)
    dif_com, dif_sem = 0.0, 0.0
    for _ in range(n_casos):
        n = int(rng.integers(5, 60))
        real = rng.random(n) < rng.uniform(0.1, 0.6)
        real[0] = True                                           # pelo menos um evento
        com, sem = rng.integers(0, 6, n).astype(float), rng.permutation(n).astype(float)
        dif_com = max(dif_com, abs(ap_com_empates(real, com) - _ap_forca_bruta(real, com)))
        dif_sem = max(dif_sem, abs(ap_do_notebook(real, sem) - _ap_forca_bruta(real, sem)), abs(ap_com_empates(real, sem) - _ap_forca_bruta(real, sem)))
    return dif_com, dif_sem


def _ri_contagens(y: np.ndarray, p: np.ndarray, limiar: float, limiar_prev: float) -> dict:
    """RI como detecção de evento (cópia de `metricas_ri` do notebook, mais as contagens): o evento REAL é ΔVmax ≥ `limiar`; o alarme é ΔVmax previsto
    ≥ `limiar_prev`. POD = TP/(TP+FN), FAR = FP/(TP+FP), CSI = TP/(TP+FN+FP); NaN onde o denominador é zero (FAR sem nenhum alarme)."""
    ok = ~(np.isnan(y) | np.isnan(p))
    real, prev = y[ok] >= limiar, p[ok] >= limiar_prev
    tp, fn, fp, tn = int((real & prev).sum()), int((real & ~prev).sum()), int((~real & prev).sum()), int((~real & ~prev).sum())
    div = lambda a, b: float(a / b) if b > 0 else np.nan
    return dict(limiar_prev=float(limiar_prev), tp=tp, fn=fn, fp=fp, tn=tn, alarmes=tp + fp, pod=div(tp, tp + fn), far=div(fp, tp + fp), csi=div(tp, tp + fn + fp))


def _limiar_ajustado(y_val: np.ndarray, p_val: np.ndarray, limiar: float) -> float:
    """Cópia de `limiar_ri_ajustado` do notebook: o limiar do ΔVmax PREVISTO (5 … `limiar` nós, passo 1) que maximiza o CSI de RI na VALIDAÇÃO
    (o teste nunca entra). Sem eventos na validação, fica o próprio `limiar`."""
    ok = ~(np.isnan(y_val) | np.isnan(p_val))
    if not (y_val[ok] >= limiar).any():
        return float(limiar)
    grade = np.arange(min(5.0, limiar), limiar + 0.5, 1.0)
    csi = [_ri_contagens(y_val, p_val, limiar, g)["csi"] for g in grade]
    return float(grade[int(np.argmax(np.nan_to_num(csi, nan=-1.0)))])


def ri_do_teste(execs: list, limiar_ri: float) -> list:
    """Uma entrada por execução COM cabeça de ΔVmax: eventos de RI no teste, o que a rede previu de ΔVmax e o quadro de alarmes e acertos com o limiar
    fixo (o próprio `limiar_ri`) e com o ajustado na validação (recalculado das previsões da validação que o `.npz` guarda)."""
    saida = []
    for e in execs:
        if not np.isfinite(e["pred_dvmax"]).any():
            continue
        y, p = e["y_dvmax"].astype(np.float64), e["pred_dvmax"].astype(np.float64)
        ok = ~(np.isnan(y) | np.isnan(p))
        real = y[ok] >= limiar_ri
        lim_aj = _limiar_ajustado(e["y_dvmax_val"].astype(np.float64), e["pred_dvmax_val"].astype(np.float64), limiar_ri)
        saida.append(dict(variante=e["variante"], seed=e["seed"], janelas=int(ok.sum()), eventos=int(real.sum()), prevalencia=float(real.mean()),
                          prev_min=float(p[ok].min()), prev_media=float(p[ok].mean()), prev_max=float(p[ok].max()),
                          prev_media_eventos=float(p[ok][real].mean()) if real.any() else np.nan, prev_media_demais=float(p[ok][~real].mean()),
                          fixo=_ri_contagens(y, p, limiar_ri, limiar_ri), ajustado=_ri_contagens(y, p, limiar_ri, lim_aj)))
    return saida


def metricas_recalculadas(execs: list, limiar_ri: float) -> pd.DataFrame:
    """Uma linha por execução, recalculada das predições do conjunto de teste (as colunas de `metricas_por_execucao.csv` que dependem só delas)."""
    linhas = []
    for e in execs:
        mv, md = erros(e["y_vmax"], e["pred_vmax"]), erros(e["y_dvmax"], e["pred_dvmax"])
        linha = dict(variante=e["variante"], seed=e["seed"], rmse_vmax=mv["rmse"], mae_vmax=mv["mae"], vies_vmax=mv["vies"],
                     rmse_dvmax=md["rmse"], mae_dvmax=md["mae"], auc_ri=np.nan, ap_ri=np.nan)
        if md["n"] > 0:
            ok = ~(np.isnan(e["y_dvmax"]) | np.isnan(e["pred_dvmax"]))
            real = e["y_dvmax"][ok] >= limiar_ri
            linha.update(auc_ri=auc_roc(real, e["pred_dvmax"][ok]), ap_ri=ap_do_notebook(real, e["pred_dvmax"][ok]))
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values(["variante", "seed"]).reset_index(drop=True)


def referencias_sem_imagens(execs: list, variante_ref: str, base: pd.DataFrame) -> dict:
    """Réguas da rodada: preditor constante em Vmax (média do treino), «prever ΔVmax = 0» e o baseline de histórico (do CSV do notebook:
    as previsões dele não ficam nos `.npz`)."""
    const, zero = [], []
    for e in execs:
        if e["variante"] != variante_ref:
            continue
        const.append(erros(e["y_vmax"], np.full_like(e["y_vmax"], e["meta"]["norm"]["vmax"][0]))["rmse"])
        zero.append(erros(e["y_dvmax"], np.zeros_like(e["y_dvmax"]))["rmse"])
    h = base[base["variante"] == "baseline_historico"]
    return dict(rmse_vmax_constante=float(np.mean(const)), rmse_dvmax_zero=float(np.mean(zero)),
                historico={c: float(h[c].mean()) for c in ("rmse_dvmax", "mae_dvmax", "auc_ri", "ap_ri")} if len(h) else {})


def conferir_ap(execs: list, base: pd.DataFrame, limiar_ri: float) -> dict:
    """AP do escore constante («prever zero») no teste: o do notebook (reproduzido aqui e no CSV dele), o com empates e a prevalência;
    e a diferença entre as duas versões nas previsões das redes."""
    e0 = next(e for e in execs if np.isfinite(e["pred_dvmax"]).any())
    ok = ~np.isnan(e0["y_dvmax"])
    real, zero = e0["y_dvmax"][ok] >= limiar_ri, np.zeros(int(ok.sum()))
    csv = base[base["variante"] == "baseline_zero"]["ap_ri"]
    teste = dict(janelas=int(ok.sum()), eventos=int(real.sum()), prevalencia=float(real.mean()), ap_notebook=ap_do_notebook(real, zero),
                 ap_csv=float(csv.iloc[0]) if len(csv) else np.nan, ap_empates=ap_com_empates(real, zero))
    redes = []
    for e in execs:
        if np.isfinite(e["pred_dvmax"]).any():
            ok2 = ~(np.isnan(e["y_dvmax"]) | np.isnan(e["pred_dvmax"]))
            r2, p = e["y_dvmax"][ok2] >= limiar_ri, e["pred_dvmax"][ok2]
            redes.append(dict(arquivo=e["arquivo"], empates=int(p.size - np.unique(p).size), dif=abs(ap_do_notebook(r2, p) - ap_com_empates(r2, p))))
    return dict(teste=teste, redes=redes)


def erro_por_faixa(execs: list, ordem: list) -> pd.DataFrame:
    """Viés, RMSE e MAE de Vmax por faixa de intensidade (Vmax real), por variante; com várias seeds, média entre elas (`n` = janelas da faixa
    numa seed). As faixas são as de `BINS_INTENSIDADE`."""
    linhas = []
    for var in ordem:
        es = [e for e in execs if e["variante"] == var]
        if not es:
            continue
        for lo, hi, nome in BINS_INTENSIDADE:
            ms = []
            for e in es:
                y, p = e["y_vmax"].astype(np.float64), e["pred_vmax"].astype(np.float64)
                faixa_ = (y >= lo) & (y < hi)
                ms.append(erros(y[faixa_], p[faixa_]))                 # já traz `n` (janelas da faixa com previsão)
            linhas.append(dict(variante=var, faixa=nome, n=float(np.mean([m["n"] for m in ms])), rmse=float(np.mean([m["rmse"] for m in ms])),
                               mae=float(np.mean([m["mae"] for m in ms])), vies=float(np.mean([m["vies"] for m in ms]))))
    return pd.DataFrame(linhas)


def erros_grandes(R: dict) -> dict:
    """Janelas com |erro de Vmax| ≥ `outlier_erro_nos` na variante de referência (1ª seed): quantas, que parcela do erro quadrático total e a RMSE sem
    elas. De quais ciclones são vem do `janelas_erro_grande_vmax.csv` do notebook (ele traz o ID de cada janela); o número de linhas dele só é
    comparável ao recalculado aqui numa rodada de 1 seed (com mais, o notebook usa o erro médio entre as seeds)."""
    cfg, exs = R["cfg"], R["exs"]
    var, seed = cfg["variante_referencia"], min(e["seed"] for e in exs)
    e = next(e for e in exs if e["variante"] == var and e["seed"] == seed)
    y, p = e["y_vmax"].astype(np.float64), e["pred_vmax"].astype(np.float64)
    d = p - y
    grande = np.abs(d) >= cfg["outlier_erro_nos"]
    csv = R["pasta"] / "tabelas" / "janelas_erro_grande_vmax.csv"
    tab = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    por_ciclone = []
    if len(tab):
        g = tab.groupby("id_ciclone").agg(n=("erro", "size"), vmax_medio=("vmax_real", "mean"), erro_medio=("erro", "mean"), de=("instante", "min"), ate=("instante", "max"))
        por_ciclone = [dict(id_ciclone=str(i), n=int(r["n"]), vmax_medio=float(r["vmax_medio"]), erro_medio=float(r["erro_medio"]), de=str(r["de"]), ate=str(r["ate"]))
                       for i, r in g.sort_values("n", ascending=False).iterrows()]
    return dict(variante=var, seed=seed, limite=float(cfg["outlier_erro_nos"]), janelas=int(len(d)), n=int(grande.sum()),
                parcela_sse=float((d[grande] ** 2).sum() / (d ** 2).sum()), rmse_com=float(np.sqrt(np.mean(d ** 2))), rmse_sem=float(np.sqrt(np.mean(d[~grande] ** 2))),
                linhas_csv=int(len(tab)) if len(tab) else None, mesmo_numero=bool(len(tab) == int(grande.sum())) if len(tab) and len({x["seed"] for x in exs}) == 1 else None,
                por_ciclone=por_ciclone)


def latentes_da_kan(pasta: Path, variante: str) -> list:
    """As 3 variáveis latentes `h_i` de `h(t)` com maior |ρ| de Spearman com cada grandeza (Vmax e ΔVmax reais; resumos do quadro `t`), da tabela
    `kan_latentes_correlacoes_<variante>.csv` do notebook (modelo da 1ª seed). Lista vazia se o arquivo não existir."""
    csv = pasta / "tabelas" / f"kan_latentes_correlacoes_{variante}.csv"
    if not csv.exists():
        return []
    c = pd.read_csv(csv, index_col=0)
    return [dict(grandeza=col, top=[(str(k), float(c.loc[k, col])) for k in c[col].abs().sort_values(ascending=False).head(3).index]) for col in c.columns]


# ----------------------------------------------------------------------------- 3. verificações de integridade
def _num(x: dict, chave: str) -> float:
    v = x.get(chave)
    return np.nan if v is None else float(v)


def verificar(R: dict) -> list:
    """Cada verificação é (nome, ok, detalhe)."""
    exs, cfg, res, pasta = R["exs"], R["cfg"], R["resumo"], R["pasta"]
    prot, v = res["protocolo"], []

    plano = len(prot["seeds"]) * len(prot["variantes"])
    v.append(("plano completo", len(exs) == plano == res["execucoes"] and res["execucoes_pendentes"] == 0,
              f"{len(exs)} execuções em `runs/` de {plano} planejadas; pendentes: {res['execucoes_pendentes']}"))

    sigs = {e["assinatura"] for e in exs}
    v.append(("mesma assinatura em todos os arquivos", sigs == {res["assinatura"]}, f"`runs/`: {sorted(sigs)}; `resumo_final.json`: {res['assinatura']}"))

    ruim = [e["arquivo"] for e in exs if e["meta"]["n_treino"] + e["meta"]["n_val"] + e["meta"]["n_teste"] != res["dados"]["janelas"]
            or len(e["idx_teste"]) != e["meta"]["n_teste"]]
    v.append(("treino + validação + teste = janelas", not ruim, f"{res['dados']['janelas']} janelas; execuções fora da regra: {len(ruim)}"))

    testes = {e["idx_teste"].tobytes() for e in exs}
    v.append(("mesmas janelas de teste em todas as execuções", len(testes) == 1,
              f"{len(exs)} execuções; conjuntos de teste distintos: {len(testes)} (a comparação entre variantes e seeds precisa do MESMO teste)"))

    reg = prot.get("janelas", {})
    m0 = exs[0]["meta"]
    iguais = (reg.get("treino"), reg.get("validacao"), reg.get("teste")) == (m0["n_treino"], m0["n_val"], m0["n_teste"])
    v.append(("divisão registrada = divisão das execuções", iguais,
              f"`resumo_final.json`: treino {reg.get('treino')} · validação {reg.get('validacao')} · teste {reg.get('teste')}; execuções: {m0['n_treino']} · {m0['n_val']} · {m0['n_teste']} ({prot.get('descricao_divisao', '?')})"))

    cols = ("rmse_vmax", "mae_vmax", "vies_vmax", "rmse_dvmax", "mae_dvmax", "auc_ri", "ap_ri")
    m = R["met_re"].merge(R["met_nb"], on=["variante", "seed"], suffixes=("", "_nb"))
    mesmos_vazios = all((m[c].isna() == m[c + "_nb"].isna()).all() for c in cols)
    dif = {c: float((m[c] - m[c + "_nb"]).abs().max()) for c in cols if (m[c] - m[c + "_nb"]).notna().any()}
    R["maior_dif_metricas"] = max(dif.values())
    v.append(("métricas recalculadas das predições = tabela do notebook", len(m) == len(exs) and mesmos_vazios and R["maior_dif_metricas"] < TOL_METRICAS,
              f"maior diferença absoluta {R['maior_dif_metricas']:.1e} (RMSE, MAE e viés de Vmax e ΔVmax, AUC e AP); tolerância {TOL_METRICAS:g}"))

    nb, ruim, dif_ri = R["met_nb"].set_index(["variante", "seed"]), [], 0.0
    for r in R["ri"]:
        linha = nb.loc[(r["variante"], r["seed"])]
        for col, val in (("limiar_ri_aj", r["ajustado"]["limiar_prev"]), ("pod_ri", r["fixo"]["pod"]), ("far_ri", r["fixo"]["far"]), ("csi_ri", r["fixo"]["csi"]),
                         ("pod_ri_aj", r["ajustado"]["pod"]), ("far_ri_aj", r["ajustado"]["far"]), ("csi_ri_aj", r["ajustado"]["csi"])):
            do_nb = float(linha[col])
            if np.isnan(val) != np.isnan(do_nb):
                ruim.append(f"{r['variante']} s{r['seed']} {col}")
            elif not np.isnan(val):
                dif_ri = max(dif_ri, abs(val - do_nb))
    R["maior_dif_ri"] = dif_ri
    v.append(("RI: limiar ajustado, POD, FAR e CSI recalculados = tabela do notebook", not ruim and dif_ri < TOL_RI,
              f"{len(R['ri'])} execuções com cabeça de ΔVmax; maior diferença absoluta {dif_ri:.1e} (o limiar ajustado é refeito da previsão da validação gravada no `.npz`; tolerância {TOL_RI:g}); "
              f"NaN diferentes: {len(ruim)}"))

    ruim = []
    for e in exs:
        h, mt = e["meta"]["historico"], e["meta"]
        rm = np.array([_num(x, "rmse_val_vmax") for x in h])
        ep_csv = int(nb.loc[(e["variante"], e["seed"]), "melhor_epoca"])
        if not (len(h) == mt["n_epocas"] <= cfg["epocas_max"] and int(np.argmin(rm)) == mt["melhor_epoca"] and ep_csv == mt["melhor_epoca"] + 1
                and abs(rm.min() - mt["rmse_val"]) < 1e-4):
            ruim.append(e["arquivo"])
    v.append(("melhor época coerente com o histórico", not ruim, f"melhor época = argmin da RMSE de validação de Vmax em todas as execuções; fora da regra: {len(ruim)}"))

    ruim = []
    for e in exs:
        tem = bool(np.isfinite(e["pred_dvmax"]).any())
        for x in e["meta"]["historico"]:
            if not all(np.isfinite(_num(x, c)) for c in COLUNAS_CURVAS + (COLUNAS_CURVAS_DVMAX if tem else ())):
                ruim.append(e["arquivo"])
            if not tem and any(np.isfinite(_num(x, c)) for c in COLUNAS_CURVAS_DVMAX):
                ruim.append(e["arquivo"])
    R["curvas_ok"] = not ruim
    v.append(("colunas novas das curvas preenchidas", not ruim,
              f"{', '.join(COLUNAS_CURVAS)} em todas as épocas; {', '.join(COLUNAS_CURVAS_DVMAX)} só nas variantes com cabeça de ΔVmax; execuções fora da regra: {len(set(ruim))}"))

    ruim = [e["arquivo"] for e in exs
            if not (np.isfinite(e["pred_vmax"]).all() and np.isfinite(e["y_vmax"]).all()
                    and (np.isfinite(e["pred_dvmax"]).all() or np.isnan(e["pred_dvmax"]).all()))]
    v.append(("predições finitas", not ruim, f"sem NaN/∞ em Vmax; ΔVmax finito ou ausente por inteiro (variante só-Vmax); fora da regra: {len(ruim)}"))

    d = res["dados"]["duplicatas"]
    obtido = {k: d[k] for k in DUPLICATAS_TCIR}
    inteiro = res["dados"]["imagens"] == IMAGENS_TCIR
    v.append(("duplicatas como no TCIR inteiro", bool(d["detectadas"]) and (obtido == DUPLICATAS_TCIR if inteiro else True),
              f"{d['registros_removidos']} registros removidos, {d['ids_unificados']} IDs unificados, {d['pares_identicos_bit_a_bit']} de {d['pares_conferidos_nos_pixels']} pares idênticos "
              + (f"(esperado: {DUPLICATAS_TCIR['registros_removidos']}, {DUPLICATAS_TCIR['ids_unificados']} e {DUPLICATAS_TCIR['pares_identicos_bit_a_bit']} de 72)" if inteiro
                 else f"(só {res['dados']['imagens']} das {IMAGENS_TCIR} imagens do TCIR: subconjunto, o esperado do TCIR inteiro não se aplica)")))

    listados = arquivos_do_relatorio(R["relatorio"])
    faltam = [f"{p}/{n}" for p, n in listados if not (pasta / p / n).exists()]
    sem_pesos = [e["arquivo"] for e in exs if e["seed"] == min(prot["seeds"]) and not (pasta / "runs" / e["arquivo"].replace(".npz", ".pt")).exists()]
    v.append(("arquivos citados no RELATORIO.md existem", not faltam and not sem_pesos and bool(listados),
              f"{len(listados)} arquivos de `figuras/` e `tabelas/` citados, faltam {len(faltam)}; execuções da 1ª seed sem `.pt`: {len(sem_pesos)}"))

    dif_com, dif_sem = autoteste_ap()
    R["autoteste_ap"] = (dif_com, dif_sem)
    v.append(("autoteste do AP com empates (função deste script)", max(dif_com, dif_sem) < 1e-12,
              f"contra a definição direta: diferença máxima {dif_com:.1e} com empates e {dif_sem:.1e} sem empates ({N_CASOS_AUTOTESTE} casos aleatórios)"))
    return v


# ----------------------------------------------------------------------------- 4. relatório
def fmt(x, casas: int = 2) -> str:
    return "—" if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) else f"{x:.{casas}f}"


def faixa(valores) -> str:
    """«a–b» (ou só «a», se todos iguais): a faixa de um número que muda de execução para execução."""
    lo, hi = min(valores), max(valores)
    return f"{lo}" if lo == hi else f"{lo}–{hi}"


def lista_de_nomes(vs: list) -> str:
    """«`a`», «`a` e `b`» ou «`a`, `b` e `c`»: nomes de variantes em texto corrido."""
    ts = [f"`{v}`" for v in vs]
    return ts[0] if len(ts) == 1 else ", ".join(ts[:-1]) + " e " + ts[-1]


def mais_menos(media, dp, casas: int = 2) -> str:
    if media is None or np.isnan(media):
        return "—"
    return f"{media:.{casas}f}" if dp is None or np.isnan(dp) else f"{media:.{casas}f} ± {dp:.{casas}f}"


def formatar_tempo(segundos: float) -> str:
    h, resto = divmod(int(segundos), 3600)
    m, s = divmod(resto, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def tabela_desempenho(met: pd.DataFrame, ordem: list) -> list:
    cols = ["rmse_vmax", "mae_vmax", "vies_vmax", "rmse_dvmax", "mae_dvmax", "auc_ri", "ap_ri"]
    g = met.groupby("variante")
    media, dp, n = g[cols].mean(), g[cols].std(ddof=1), g.size()
    linhas = ["| variante | execuções (seeds) | Vmax RMSE | Vmax MAE | Vmax viés | ΔVmax RMSE | ΔVmax MAE | RI AUC | RI AP |", "|---|---|---|---|---|---|---|---|---|"]
    for var in ordem:
        if var in media.index:
            linhas.append(f"| {var} | {n[var]} | " + " | ".join(mais_menos(media.loc[var, c], dp.loc[var, c], 3 if c in ("auc_ri", "ap_ri") else 2) for c in cols) + " |")
    return linhas


def tabela_por_seed(met: pd.DataFrame, col: str, ordem: list) -> list:
    """Uma coluna de `met` (p. ex. a RMSE) por seed, uma linha por variante. Só faz sentido com mais de uma seed."""
    g = met.pivot_table(index="variante", columns="seed", values=col, aggfunc="mean")
    seeds = list(g.columns)
    linhas = ["| variante | " + " | ".join(f"seed {s}" for s in seeds) + " | média ± dp |", "|---|" + "---|" * (len(seeds) + 1)]
    for var in ordem:
        if var in g.index and g.loc[var].notna().any():
            linhas.append(f"| {var} | " + " | ".join(fmt(g.loc[var, s]) for s in seeds) + f" | {mais_menos(g.loc[var].mean(), g.loc[var].std(ddof=1))} |")
    return linhas


def tabela_curvas(execs: list, ordem: list) -> list:
    linhas = ["| variante | melhor época (média) | épocas rodadas | RMSE de Vmax: treino (modo avaliação) | RMSE de Vmax: validação | folga (validação − treino) | segundos por época (mediana) |",
              "|---|---|---|---|---|---|---|"]
    for var in ordem:
        es = [e for e in execs if e["variante"] == var]
        if not es:
            continue
        melhor = [e["meta"]["historico"][e["meta"]["melhor_epoca"]] for e in es]
        tr, va = np.mean([_num(h, "rmse_treino_vmax") for h in melhor]), np.mean([_num(h, "rmse_val_vmax") for h in melhor])
        seg = np.median([_num(h, "segundos") for e in es for h in e["meta"]["historico"]])
        linhas.append(f"| {var} | {np.mean([e['meta']['melhor_epoca'] + 1 for e in es]):.1f} | {np.mean([e['meta']['n_epocas'] for e in es]):.1f} | "
                      f"{tr:.2f} | {va:.2f} | {va - tr:+.2f} | {seg:.2f} |")
    return linhas


def tabela_curvas_por_epoca(execs: list, ordem: list) -> list:
    """RMSE de treino (modo avaliação) / de validação e a folga (validação − treino) em algumas épocas: média das execuções da variante que chegaram a elas
    (o *early stopping* encerra algumas antes). Uma linha por (variante, tarefa); ΔVmax só nas variantes com a cabeça."""
    ks = [k for k in EPOCAS_CURVA_TREINO if any(len(e["meta"]["historico"]) >= k for e in execs)]
    linhas = ["| variante | tarefa | " + " | ".join(f"época {k}" for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
    for var in ordem:
        es = [e for e in execs if e["variante"] == var]
        for tarefa, c_tr, c_va in (("Vmax", "rmse_treino_vmax", "rmse_val_vmax"), ("ΔVmax", "rmse_treino_dvmax", "rmse_val_dvmax")):
            celulas = []
            for k in ks:
                hs = [e["meta"]["historico"][k - 1] for e in es if len(e["meta"]["historico"]) >= k]
                tr, va = [_num(h, c_tr) for h in hs], [_num(h, c_va) for h in hs]
                celulas.append(f"{np.mean(tr):.1f} / {np.mean(va):.1f} ({np.mean(va) - np.mean(tr):+.1f})" if hs and np.isfinite(tr).all() and np.isfinite(va).all() else "—")
            if any(c != "—" for c in celulas):
                linhas.append(f"| {var} | {tarefa} | " + " | ".join(celulas) + " |")
    return linhas


def tabela_dvmax_melhor_ponto(execs: list, ordem: list) -> list:
    """ΔVmax na época que o *early stopping* guardou (escolhida pela RMSE de validação de VMAX) e na época em que a validação de ΔVmax foi menor. O mínimo de
    ≈ 25 valores é otimista: o ganho é um TETO do que escolher a época pelo ΔVmax daria na validação; não dá para conferi-lo no teste (as predições do teste
    só existem para a época escolhida pelo Vmax). Vazia se nenhuma variante tem a cabeça de ΔVmax."""
    linhas = ["| variante | ΔVmax treino (modo avaliação) | ΔVmax validação | folga | melhor época de ΔVmax na validação (média) | validação nela | ganho possível na validação |",
              "|---|---|---|---|---|---|---|"]
    for var in ordem:
        es = [e for e in execs if e["variante"] == var and np.isfinite(_num(e["meta"]["historico"][0], "rmse_val_dvmax"))]
        if not es:
            continue
        guardado = [e["meta"]["historico"][e["meta"]["melhor_epoca"]] for e in es]
        tr, va = np.mean([_num(h, "rmse_treino_dvmax") for h in guardado]), np.mean([_num(h, "rmse_val_dvmax") for h in guardado])
        curvas = [np.array([_num(h, "rmse_val_dvmax") for h in e["meta"]["historico"]]) for e in es]
        melhor_va, melhor_ep = np.mean([c.min() for c in curvas]), np.mean([int(np.argmin(c)) + 1 for c in curvas])
        linhas.append(f"| {var} | {tr:.2f} | {va:.2f} | {va - tr:+.2f} | {melhor_ep:.1f} | {melhor_va:.2f} | {va - melhor_va:.2f} |")
    return linhas if len(linhas) > 2 else []


def folga_no_melhor(execs: list, variante: str) -> dict:
    """RMSE de treino (modo avaliação) e de validação, de Vmax e de ΔVmax, na época guardada (média das execuções da variante); NaN onde a variante não tem a coluna."""
    hs = [e["meta"]["historico"][e["meta"]["melhor_epoca"]] for e in execs if e["variante"] == variante]
    media = lambda c: float(np.mean([_num(h, c) for h in hs])) if hs else np.nan
    return dict(vmax_tr=media("rmse_treino_vmax"), vmax_va=media("rmse_val_vmax"), dvmax_tr=media("rmse_treino_dvmax"), dvmax_va=media("rmse_val_dvmax"))


def tabela_faixas(faixas: pd.DataFrame, ordem: list) -> list:
    """Uma linha por faixa de intensidade; por variante, «viés / RMSE» de Vmax (nós)."""
    vs = [v for v in ordem if v in set(faixas["variante"])]
    linhas = ["| faixa (Vmax real) | janelas | " + " | ".join(f"{v}: viés / RMSE" for v in vs) + " |", "|---|---|" + "---|" * len(vs)]
    for _, _, nome in BINS_INTENSIDADE:
        g = faixas[faixas["faixa"] == nome]
        celulas = []
        for v in vs:
            r = g[g["variante"] == v]
            celulas.append(f"{r['vies'].iloc[0]:+.1f} / {r['rmse'].iloc[0]:.1f}" if len(r) and np.isfinite(r["rmse"].iloc[0]) else "—")
        linhas.append(f"| {nome} | {g['n'].iloc[0]:.0f} | " + " | ".join(celulas) + " |")
    return linhas


def tabela_tcc(met: pd.DataFrame, ordem: list, n_teste: int) -> list:
    """O desempenho no teste ao lado do TCC (mesma divisão por ano: treino 2003–2014, validação 2015–2016, teste 2017). Ressalvas: aqui a amostra é uma janela
    de 5 quadros (≈ 4 mil janelas de teste, contra 4296 imagens no TCC) e a entrada é outra (4 canais por quadro, em sequência)."""
    linhas = ["| modelo | RMSE (nós) | MAE (nós) | N de teste |", "|---|---|---|---|"]
    g = met.groupby("variante")[["rmse_vmax", "mae_vmax"]].mean()
    for var in ordem:
        if var in g.index:
            linhas.append(f"| {var} (esta rodada) | {g.loc[var, 'rmse_vmax']:.2f} | {g.loc[var, 'mae_vmax']:.2f} | {n_teste} janelas |")
    linhas += [f"| {nome} | {rmse:.2f} | {mae:.2f} | 4296 imagens |" for nome, (rmse, mae) in REF_TCC.items()]
    return linhas


def bloco_ri(R: dict) -> list:
    """RI no teste: o que a rede previu de ΔVmax, o quadro de alarmes e acertos com o limiar fixo e com o ajustado na validação e, ao lado, o baseline de
    histórico (POD, FAR e CSI do CSV do notebook: as previsões dele não ficam nos `.npz`, então não há contagens). Vazio se nenhuma variante tem a cabeça de ΔVmax."""
    cfg, ri = R["cfg"], R["ri"]
    if not ri:
        return []
    lim = cfg["limiar_ri_nos"]
    L = [f"**RI (ΔVmax ≥ {lim:.0f} nós em {cfg['horizonte_horas']} h) no teste.** O modelo não prevê RI: prevê ΔVmax em nós, e a RI é o evento «ΔVmax real ≥ {lim:.0f}». "
         f"O alarme é «ΔVmax previsto ≥ limiar», com o limiar fixo ({lim:.0f}) ou com um ajustado na validação (o que maximiza o CSI lá; o teste não entra). AUC e AP não usam limiar: "
         "só a ordem dos escores.", "",
         "O que a rede previu de ΔVmax (nós):", "",
         "| variante · seed | janelas com ΔVmax | eventos de RI (prevalência) | previsto: mín / média / máx | previsto médio nos eventos × nos demais |", "|---|---|---|---|---|"]
    L += [f"| {r['variante']} · {r['seed']} | {r['janelas']} | {r['eventos']} ({r['prevalencia']:.3f}) | {r['prev_min']:.1f} / {r['prev_media']:.1f} / {r['prev_max']:.1f} | "
          f"{fmt(r['prev_media_eventos'], 1)} × {r['prev_media_demais']:.1f} |" for r in ri]
    L += ["", "Alarmes e acertos (POD = acertos ÷ eventos reais; FAR = alarmes falsos ÷ alarmes; CSI = acertos ÷ (acertos + perdidos + falsos)):", "",
          "| modelo | limiar de alarme (nós) | alarmes | acertos (TP) | alarmes falsos (FP) | eventos perdidos (FN) | POD | FAR | CSI |", "|---|---|---|---|---|---|---|---|---|"]
    for r in ri:
        for rotulo, c in (("fixo", r["fixo"]), ("ajustado na validação", r["ajustado"])):
            L.append(f"| {r['variante']} · {r['seed']} | {c['limiar_prev']:.0f} ({rotulo}) | {c['alarmes']} | {c['tp']} | {c['fp']} | {c['fn']} | {fmt(c['pod'])} | {fmt(c['far'])} | {fmt(c['csi'])} |")
    h = R["base"][R["base"]["variante"] == "baseline_historico"]
    if len(h):
        m = h.mean(numeric_only=True)
        for rotulo, lim_h, pod, far, csi in (("fixo", lim, m["pod_ri"], m["far_ri"], m["csi_ri"]),
                                             ("ajustado no treino", m["limiar_ri_aj"], m["pod_ri_aj"], m["far_ri_aj"], m["csi_ri_aj"])):
            L.append(f"| baseline_historico (CSV do notebook) | {lim_h:.0f} ({rotulo}) | — | — | — | — | {fmt(pod)} | {fmt(far)} | {fmt(csi)} |")
    L += ["", "Os baselines «zero» e «média» têm previsão constante abaixo do limiar: nunca dão alarme (POD 0, FAR indefinido). "
              "Um limiar ajustado abaixo de 30 compensa o encolhimento das previsões em direção à média (a regressão por erro quadrático quase nunca prevê um ΔVmax tão alto).", ""]
    return L


def montar_relatorio(R: dict) -> list:
    cfg, res, exs, ver = R["cfg"], R["resumo"], R["exs"], R["verificacoes"]
    teste, nome, prot, d = cfg["modo"] == "teste", res["experimento"]["nome"], res["protocolo"], res["dados"]
    ordem = cfg["variantes"]
    seeds = ", ".join(str(s) for s in prot["seeds"])
    n_ok, falharam = sum(1 for _, ok, _ in ver if ok), [n for n, ok, _ in ver if not ok]
    n_fig, n_tab = len(list((R["pasta"] / "figuras").glob("*.png"))), len(list((R["pasta"] / "tabelas").glob("*.csv")))
    mt = [e["meta"] for e in exs]
    ntr, nval, nte = (sorted({m[k] for m in mt}) for k in ("n_treino", "n_val", "n_teste"))
    refs, ap = R["refs"], R["ap"]
    boot = R["bootstrap"]
    por_ano = prot.get("divisao") == "ano"
    por_var = R["met_re"].groupby("variante")
    redes_vmax = por_var["rmse_vmax"].mean()
    redes_dv = por_var["rmse_dvmax"].mean().dropna()
    L = [f"# Rodada «{nome}» (seed {seeds}) do notebook rápido — modo {cfg['modo']}", "",
         f"Gerado por `analisar_rodada_rapida.py` a partir de `{R['entrada']}` (rodada gerada em {R['gerado_em']}). Erros em nós.", ""]
    if teste:
        L += [f"> ⚠ **MODO TESTE** ({cfg['max_ciclones']} ciclones, {cfg['epocas_max']} épocas, {cfg['bootstrap_reps']} réplicas de bootstrap): "
              "os números **não são resultado do experimento**. O que esta rodada mostra é se o notebook funciona de ponta a ponta na GPU do Kaggle.", ""]

    # --- 1. resumo
    plano = len(prot["seeds"]) * len(prot["variantes"])
    aprendem = bool(redes_vmax.max() < refs["rmse_vmax_constante"])
    def valor_ou_faixa(v) -> str:       # «é 8.39» com uma variante só; «entre 8.39 e 9.10» com várias
        lo, hi = f"{v.min():.2f}", f"{v.max():.2f}"
        return f"é {lo}" if lo == hi else f"entre {lo} e {hi}"
    numeros = (f"RMSE de Vmax no teste {valor_ou_faixa(redes_vmax)} (preditor constante: {refs['rmse_vmax_constante']:.1f})"
               + (f"; RMSE de ΔVmax {valor_ou_faixa(redes_dv)} («prever zero»: {refs['rmse_dvmax_zero']:.1f})" if len(redes_dv) else "") + ".")
    if teste:
        numeros += (f" Com {cfg['epocas_max']} épocas e {faixa(ntr)} janelas de treino, em Vmax as redes " + ("ficam bem abaixo" if aprendem else "NÃO ficam abaixo")
                    + " do preditor constante, mas é só isso que o teste mostra.")
    topicos = [
        f"**Rodou de ponta a ponta** (dispositivo `{R['dispositivo']}`): {len(exs)} de {plano} treinos, {res['execucoes_pendentes']} pendentes, {n_fig} figuras, {n_tab} tabelas e o `RELATORIO.md`.",
        f"**Integridade:** {n_ok} de {len(ver)} verificações passaram (seção 3)" + (f"; falharam: {', '.join(falharam)}" if falharam else "")
        + f". Recalculadas das predições, as métricas diferem das tabelas do notebook em no máximo {R['maior_dif_metricas']:.1e} nó.",
        "**Divisão:** " + prot.get("descricao_divisao", "?") + f" — treino {faixa(ntr)} janelas, validação {faixa(nval)}, teste {faixa(nte)}.",
        "**Curvas de treino:** " + ("as colunas novas (perda e RMSE de treino em modo de avaliação; validação de ΔVmax) vieram preenchidas em todas as épocas; "
                                    "as de ΔVmax ficam vazias só na variante só-Vmax, como devem." if R["curvas_ok"] else "há colunas novas vazias ou com NaN (seção 3)."),
        f"**Números** (seção 4): {numeros}"]
    if R["ri"]:
        r0 = next((r for r in R["ri"] if r["variante"] == cfg["variante_referencia"]), R["ri"][0])
        fx, aj = r0["fixo"], r0["ajustado"]
        topicos.append(f"**RI** (seção 4): {r0['eventos']} eventos em {r0['janelas']} janelas do teste (prevalência {r0['prevalencia']:.3f}); a `{r0['variante']}` previu ΔVmax entre "
                       f"{r0['prev_min']:.1f} e {r0['prev_max']:.1f} nós; com o limiar de {cfg['limiar_ri_nos']:.0f} nós deu {fx['alarmes']} alarmes e acertou {fx['tp']} eventos "
                       f"(POD {fmt(fx['pod'])}, FAR {fmt(fx['far'])}); com o limiar ajustado na validação ({aj['limiar_prev']:.0f} nós), {aj['alarmes']} alarmes e {aj['tp']} acertos "
                       f"(POD {fmt(aj['pod'])}, FAR {fmt(aj['far'])}, CSI {fmt(aj['csi'])}).")
    if not teste:
        if por_ano:
            v0 = cfg["variante_referencia"]
            topicos.append(f"**Contra o TCC** (mesma divisão: treino 2003–2014, teste 2017): a `{v0}` tem RMSE {redes_vmax[v0]:.2f} e MAE {por_var['mae_vmax'].mean()[v0]:.2f} nó no teste, "
                           f"contra 10,06 / 7,53 do CNN-TC e 9,14 / 7,00 do MobileNetV2+SDTA do TCC (seção 4; ressalvas lá).")
        v0 = cfg["variante_referencia"]
        so_vmax = [v for v in ordem if v != v0 and any(e["variante"] == v for e in exs) and not np.isfinite(folga_no_melhor(exs, v)["dvmax_va"])]
        f0 = folga_no_melhor(exs, v0)
        txt = (f"**Sobreajuste** (seção 5): na época guardada, a `{v0}` erra {f0['vmax_va']:.2f} nós de Vmax na validação e {f0['vmax_tr']:.2f} no treino em modo de avaliação "
               f"(folga {f0['vmax_va'] - f0['vmax_tr']:+.2f})")
        if np.isfinite(f0["dvmax_va"]):
            txt += f"; em ΔVmax, {f0['dvmax_va']:.2f} contra {f0['dvmax_tr']:.2f} (folga {f0['dvmax_va'] - f0['dvmax_tr']:+.2f})"
        for v in so_vmax:
            f1 = folga_no_melhor(exs, v)
            txt += f"; a `{v}`, {f1['vmax_va']:.2f} contra {f1['vmax_tr']:.2f} em Vmax (folga {f1['vmax_va'] - f1['vmax_tr']:+.2f})"
        topicos.append(txt + ".")
        g = R["grandes"]
        txt = (f"**Erros grandes** (seção 7): {g['n']} janelas com |erro de Vmax| ≥ {g['limite']:.0f} nós na `{g['variante']}` ({100 * g['n'] / g['janelas']:.2f}% das janelas) "
               f"respondem por {100 * g['parcela_sse']:.1f}% do erro quadrático")
        if g["por_ciclone"]:
            txt += f"; o ciclone com mais janelas entre as listadas pelo notebook é o `{g['por_ciclone'][0]['id_ciclone']}` ({g['por_ciclone'][0]['n']} de {g['linhas_csv']})"
        topicos.append(txt + ".")
    t0 = ap["teste"]
    topicos.append("**Um defeito conhecido** (seção 6): a `precisao_media` não trata empates e o AP de um escore constante sai "
                   f"{t0['ap_notebook']:.3f}, quando a prevalência de RI no teste é {t0['prevalencia']:.3f}. Afeta as linhas `baseline_zero` e `baseline_media`; "
                   "as previsões das redes não têm empates.")
    L += ["## 1. Resumo", ""] + [f"- {t}" for t in topicos] + [""]

    # --- 2. a rodada
    pars = ", ".join(f"`{var}` {R['parametros'][var]}" for var in ordem if var in R["parametros"])
    ciclones = prot.get("ciclones", {})
    # os ciclones que têm janela são os dos três conjuntos; o `resumo_final.json` também traz o total com imagem (maior: ciclones curtos não geram janela)
    n_cic_janelas = sum(v for v in (ciclones.get(k) for k in ("treino", "validacao", "teste")) if isinstance(v, int))
    L += ["## 2. A rodada", "", "| item | valor |", "|---|---|",
          f"| modo | `{cfg['modo']}`" + (" ⚠" if teste else "") + " |",
          f"| experimento · seed | `{nome}` · {seeds} |",
          f"| dispositivo | `{R['dispositivo']}` |",
          f"| assinatura | `{res['assinatura']}` |",
          f"| plano | {len(exs)} treinos = {len(prot['seeds'])} seed(s) {prot['seeds']} × {len(prot['variantes'])} variante(s); até {cfg['epocas_max']} épocas (paciência {cfg['paciencia']}) |",
          f"| variantes | " + "; ".join(f"`{v}`: {DESCRICAO_VARIANTE.get(v, 'sem descrição')}" for v in ordem) + " |",
          f"| dados | {d['imagens']} imagens ({d['ciclones']} ciclones depois de unir os IDs) → {d['janelas']} janelas"
          + (f" ({n_cic_janelas} ciclones com janela)" if n_cic_janelas else "") + f"; {d['quadros_invalidos']} quadros inválidos |",
          f"| duplicatas | {d['duplicatas']['registros_removidos']} registros removidos, {d['duplicatas']['ids_unificados']} IDs unificados; "
          f"{d['duplicatas']['pares_identicos_bit_a_bit']} de {d['duplicatas']['pares_conferidos_nos_pixels']} pares idênticos nos pixels |",
          f"| divisão | {prot.get('descricao_divisao', '?')} |",
          f"| janelas (ciclones) | treino {faixa(ntr)} ({ciclones.get('treino', '?')}) · validação {faixa(nval)} ({ciclones.get('validacao', '?')}) · teste {faixa(nte)} ({ciclones.get('teste', '?')}) |",
          f"| tempo | treino somado {R['met_nb']['segundos'].sum():.0f} s ({formatar_tempo(R['met_nb']['segundos'].sum())})"
          + (f"; do `config.json` ao `RELATORIO.md` {R['minutos']:.1f} min (carimbos dos arquivos)" if R["minutos"] else "")
          + (f": {R['fases']['preparo']:.1f} min de preparo (dados, duplicatas, janelas, cache) até a 1ª figura, {R['fases']['treino']:.1f} min até o último treino e "
             f"{R['fases']['depois']:.1f} min de análise depois dele" if R["fases"] else "") + " |",
          f"| parâmetros | {pars} |", ""]

    # --- 3. verificações
    L += ["## 3. Verificações", "", "| verificação | resultado | detalhe |", "|---|---|---|"]
    L += [f"| {n} | {'✔' if ok else '✖'} | {det} |" for n, ok, det in ver] + [""]

    # --- 4. números
    h = refs["historico"]
    L += ["## 4. Números" + (" (sem valor como resultado)" if teste else ""), "",
          f"No conjunto de **teste**. Média ± desvio entre as seeds da variante (com 1 seed só aparece o valor), recalculadas das predições (RI: ΔVmax ≥ {cfg['limiar_ri_nos']:.0f} nós "
          f"em {cfg['horizonte_horas']} h; AUC e AP com o ΔVmax previsto como escore).", ""]
    L += tabela_desempenho(R["met_re"], ordem) + [""]
    if len(prot["seeds"]) > 1:
        L += ["RMSE de Vmax por seed (nós):", ""] + tabela_por_seed(R["met_re"], "rmse_vmax", ordem) + [""]
        L += ["RMSE de ΔVmax por seed (nós):", ""] + tabela_por_seed(R["met_re"], "rmse_dvmax", ordem) + [""]
    L += ["Réguas sem imagens (no mesmo teste):", "",
          f"- preditor constante em Vmax (média do treino): RMSE {refs['rmse_vmax_constante']:.2f};",
          f"- «prever ΔVmax = 0»: RMSE {refs['rmse_dvmax_zero']:.2f};"]
    if h:
        L += [f"- baseline de histórico (do CSV do notebook): ΔVmax RMSE {h['rmse_dvmax']:.2f}, MAE {h['mae_dvmax']:.2f}, AUC {h['auc_ri']:.3f}, AP {h['ap_ri']:.3f}."]
    L += [""]
    if len(boot):
        p_min = 2 / (cfg["bootstrap_reps"] + 1)
        piso = min(1.0, p_min * len(boot))
        L += [f"Bootstrap por ciclone ({int(boot['ciclones'].iloc[0])} ciclones do teste): com {cfg['bootstrap_reps']} réplicas o menor p possível é {p_min:.4f} (= 2/({cfg['bootstrap_reps']}+1)) "
              f"e o menor observado foi {boot['p_boot'].min():.4f}; com Holm sobre {len(boot)} testes, o menor p ajustado possível é {piso:.3f}"
              + (f", acima de α = {cfg['alfa']}: todos os vereditos saem «sem diferença», qualquer que seja o tamanho do efeito." if piso > cfg["alfa"] else f" (α = {cfg['alfa']})."), ""]
        b = boot[["comparacao", "tarefa", "metrica", "dif", "ic95_inf", "ic95_sup", "p_holm", "veredito"]]
        L += ["| comparação (referência − outra) | tarefa | métrica | dif [IC 95%] | p_holm | veredito |", "|---|---|---|---|---|---|"]
        L += [f"| {r.comparacao} | {r.tarefa} | {r.metrica} | {r.dif:+.3f} [{r.ic95_inf:+.3f}; {r.ic95_sup:+.3f}] | {r.p_holm:.3f} | {r.veredito} |" for r in b.itertuples()] + [""]
    L += bloco_ri(R)
    if not teste and por_ano:
        L += ["**Contra o TCC.** Mesma divisão do TCC (treino 2003–2014, validação 2015–2016, teste 2017). Ressalvas: aqui cada amostra é uma janela de 5 quadros que exige 7 quadros consecutivos "
              "válidos (≈ 4 mil janelas de teste, e não as 4296 imagens do TCC) e a entrada tem 4 canais por quadro (IR, PMW, DTA e SDTA), contra uma imagem por vez no TCC; cada modelo foi treinado "
              "uma vez (aqui, com 1 seed: o ruído de seed, ≈ 0,45 nó entre duas rodadas, não é medido):", ""]
        L += tabela_tcc(R["met_re"], ordem, faixa(nte)) + [""]

    # --- 5. curvas
    n_aval = cfg["n_janelas_aval_treino"]
    lotes = sorted({n // cfg["tam_lote"] for n in ntr})
    L += ["## 5. Curvas de treino", "",
          "RMSE de Vmax na melhor época (a que o *early stopping* guarda); o treino é medido em modo de avaliação (sem dropout nem rotação).", ""]
    L += tabela_curvas(exs, ordem) + [""]
    L += ["Treino (modo avaliação) / validação (folga = validação − treino) ao longo das épocas; RMSE em nós, média das execuções que chegaram à época:", ""]
    L += tabela_curvas_por_epoca(exs, ordem) + [""]
    dv_tab = tabela_dvmax_melhor_ponto(exs, ordem)
    if dv_tab:
        L += ["ΔVmax na época guardada (escolhida pela validação de Vmax) e na melhor época de ΔVmax. O «ganho possível» é um teto otimista (o mínimo de várias épocas, só na validação):", ""] + dv_tab + [""]
    subconj = "todas as" if max(ntr) <= n_aval else f"um subconjunto fixo de {n_aval} das"
    passos = faixa([l * cfg["epocas_max"] for l in lotes])
    if teste:
        L += [f"- A validação tem {faixa(nval)} janelas e o treino faz só {passos} passos de otimização ({cfg['epocas_max']} épocas de {faixa(lotes)} lotes de {cfg['tam_lote']}): "
              "folga negativa, ou positiva, não diz nada sobre sobreajuste aqui."]
    else:
        L += [f"- Folga positiva = a rede erra mais em ciclones que não viu do que nas janelas de treino. Escala: validação de {faixa(nval)} janelas e até {passos} passos de "
              f"otimização ({cfg['epocas_max']} épocas de {faixa(lotes)} lotes de {cfg['tam_lote']}): aqui a folga mede sobreajuste."]
    L += [f"- O diagnóstico de treino usou {subconj} janelas de treino ({faixa(ntr)} por execução), em modo de avaliação (sem dropout nem rotação), o mesmo modo em que o teste é previsto.", ""]

    # --- 6. AP
    L += ["## 6. Defeito conhecido: AP (precisão média) com escores empatados", "",
          "A `precisao_media` (nos dois notebooks, seção 5.2, e em `analisar_resultados_pipeline.py`) ordena com `np.argsort(-escore, kind=\"stable\")`: escores iguais ficam na ordem das janelas. "
          "Um escore constante (os baselines «zero» e «média») deveria dar AP = prevalência; dá um valor que depende dessa ordem.", "",
          "| janelas com ΔVmax | eventos de RI | prevalência (= AP correto) | AP «zero» do notebook (reproduzido aqui) | AP «zero» no CSV do notebook | AP «zero» com empates |", "|---|---|---|---|---|---|",
          f"| {t0['janelas']} | {t0['eventos']} | {t0['prevalencia']:.4f} | {t0['ap_notebook']:.4f} | {t0['ap_csv']:.4f} | {t0['ap_empates']:.4f} |", ""]
    L += [f"- **Efeito:** só em escores com empates. As previsões de ΔVmax das redes são todas distintas ({sum(r['empates'] for r in ap['redes'])} empates em {len(ap['redes'])} execuções) e o AP delas é igual nas duas versões "
          f"(diferença máxima {max(r['dif'] for r in ap['redes']):.1e}). O baseline de histórico não pôde ser conferido: suas previsões não ficam nos `.npz` (e ele parte de Vmax em múltiplos de 5 nós, então pode ter empates).",
          "- **Correção proposta** (não aplicada; não exige retreinar, porque o AP é recalculado das predições): trocar a função por esta, que é a que o script usa.", "", "```python"]
    L += inspect.getsource(ap_com_empates).rstrip().splitlines() + ["```", ""]

    # --- 7. erro por faixa, erros grandes, latentes (só rodada completa)
    if teste:
        L += ["## 7. O que a rodada não mostra", "",
              f"- **Desempenho:** {cfg['epocas_max']} épocas, {faixa(ntr)} janelas de treino; nenhuma comparação entre variantes vale.",
              f"- **Curvas longas:** as figuras `10_*` e `11_*` saíram com {cfg['epocas_max']} pontos por curva; o aspecto com dezenas de épocas só se verá na rodada completa.",
              "- **Log do notebook:** o zip não traz o log (avisos, tempo de cada etapa); o tempo total acima vem só dos carimbos dos arquivos.", ""]
    else:
        g = R["grandes"]
        L += ["## 7. Erro por faixa de intensidade e erros grandes", "",
              "Viés (previsto − real) e RMSE de Vmax por faixa de intensidade (Vmax real; conjunto de teste" + ("; média entre as seeds" if len(prot["seeds"]) > 1 else "") + "), em nós:", ""]
        L += tabela_faixas(R["faixas"], ordem) + [""]
        L += [f"**Janelas com |erro de Vmax| ≥ {g['limite']:.0f} nós** (`{g['variante']}`, seed {g['seed']}): {g['n']} de {g['janelas']} ({100 * g['n'] / g['janelas']:.2f}%), "
              f"{100 * g['parcela_sse']:.1f}% do erro quadrático total; a RMSE vai de {g['rmse_com']:.2f} para {g['rmse_sem']:.2f} nós sem elas.", ""]
        if g["linhas_csv"] is not None:
            mesmo = ("o mesmo número" if g["mesmo_numero"] else "número diferente do recalculado aqui") if g["mesmo_numero"] is not None else "com mais de 1 seed o notebook usa o erro médio entre as seeds: não comparável"
            L += [f"O `janelas_erro_grande_vmax.csv` do notebook lista {g['linhas_csv']} janelas ({mesmo}). Ciclones dessas janelas (os 6 com mais):", "",
                  "| ciclone | janelas | Vmax real médio | erro médio (previsto − real) | primeira | última |", "|---|---|---|---|---|---|"]
            L += [f"| `{c['id_ciclone']}` | {c['n']} | {c['vmax_medio']:.1f} | {c['erro_medio']:+.1f} | {c['de']} | {c['ate']} |" for c in g["por_ciclone"][:6]] + [""]
            if g["por_ciclone"] and g["por_ciclone"][0]["id_ciclone"] == "201407E":
                L += ["O `201407E` é a faixa horizontal do gráfico predito × real (`figuras/02_predito_vs_real.png`): a longitude dele no TCIR vem com o sinal trocado depois da linha de data "
                      "e o recorte deixa de conter o ciclone (análise em `analisar_resultados_pipeline.py`).", ""]
        if R["latentes"]:
            L += [f"**Variáveis latentes da KAN** (`h_i` de `h(t)`; modelo da 1ª seed da `{cfg['variante_referencia']}`): as 3 com maior |ρ| de Spearman com cada grandeza "
                  "(`kan_latentes_correlacoes_*.csv` do notebook):", "", "| grandeza | 1ª | 2ª | 3ª |", "|---|---|---|---|"]
            L += [f"| {x['grandeza']} | " + " | ".join(f"{k} ({r:+.2f})" for k, r in x["top"]) + " |" for x in R["latentes"]]
            L += ["", "Uma `h_i` é uma coordenada arbitrária do espaço da LSTM (girar o espaço daria outra rede igualmente boa): correlação alta com Vmax diz que a variável codifica "
                      "intensidade, mas não lhe dá nome físico.", ""]
        L += ["## 8. O que a rodada não mostra", "",
              "- **Ruído de seed:** com 1 seed não se mede (com LSTM, nem a mesma seed repete o treino na GPU); só leve a sério diferenças acima de ≈ 0,45 nó em Vmax e ≈ 0,67 em ΔVmax entre duas rodadas.",
              "- **Variação entre conjuntos de ciclones:** o teste é um só (sem folds): o resultado depende de quais ciclones caíram nele.",
              "- **O log do notebook** (esparsidade da KAN, avisos): o zip não o traz.", ""]

    # --- 9. arquivos
    L += ["## 9. Arquivos", "", f"- Pasta da rodada: `{R['entrada']}` (`figuras/`, `tabelas/`, `runs/`, `RELATORIO.md`).",
          "- Deste script: `relatorio.md`, `resumo.json` e `verificacoes.csv`, na mesma pasta que este relatório.", ""]
    return L


# ----------------------------------------------------------------------------- 5. principal
def main() -> None:
    raiz = Path(__file__).resolve().parents[2]   # raiz do projeto (este arquivo fica em scripts/analise/)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--entrada", required=True, help="a pasta da rodada (a que está dentro do zip: <nome>_seed<k>)")
    ap.add_argument("--saida", default=str(raiz / "resultados_analise_rapido"))
    a = ap.parse_args()

    pasta = Path(a.entrada)
    cfg = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
    resumo = json.loads((pasta / "resumo_final.json").read_text(encoding="utf-8"))
    relatorio = (pasta / "RELATORIO.md").read_text(encoding="utf-8")
    exs = ler_execucoes(pasta / "runs")
    if not exs:
        raise SystemExit(f"nenhum `.npz` em {pasta / 'runs'}")
    base = pd.read_csv(pasta / "tabelas" / "baselines_dvmax.csv")
    R = dict(pasta=pasta, entrada=caminho_relativo(pasta, raiz), cfg=cfg, resumo=resumo, relatorio=relatorio, exs=exs, base=base,
             met_nb=pd.read_csv(pasta / "tabelas" / "metricas_por_execucao.csv"),
             met_re=metricas_recalculadas(exs, cfg["limiar_ri_nos"]),
             bootstrap=pd.read_csv(pasta / "tabelas" / "bootstrap_por_ciclone.csv") if (pasta / "tabelas" / "bootstrap_por_ciclone.csv").exists() else pd.DataFrame(),
             minutos=duracao_do_notebook(pasta), fases=linha_do_tempo(pasta), parametros={e["variante"]: e["meta"]["parametros"] for e in exs},
             **dados_do_relatorio(relatorio))
    R["ri"] = ri_do_teste(exs, cfg["limiar_ri_nos"])
    R["verificacoes"] = verificar(R)
    R["refs"] = referencias_sem_imagens(exs, cfg["variante_referencia"], base)
    R["ap"] = conferir_ap(exs, base, cfg["limiar_ri_nos"])
    if cfg["modo"] != "teste":
        R["faixas"], R["grandes"] = erro_por_faixa(exs, cfg["variantes"]), erros_grandes(R)
        R["latentes"] = latentes_da_kan(pasta, cfg["variante_referencia"])
    texto = montar_relatorio(R)

    saida = Path(a.saida) / pasta.name
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "relatorio.md").write_text("\n".join(texto) + "\n", encoding="utf-8")
    pd.DataFrame(R["verificacoes"], columns=["verificacao", "ok", "detalhe"]).to_csv(saida / "verificacoes.csv", index=False)
    em_json = lambda x: x.to_dict("records") if isinstance(x, pd.DataFrame) else x
    resumo_json = dict(rodada=pasta.name, modo=cfg["modo"], dispositivo=R["dispositivo"], assinatura=resumo["assinatura"], divisao=resumo["protocolo"].get("descricao_divisao"),
                       verificacoes={n: bool(ok) for n, ok, _ in R["verificacoes"]}, maior_dif_metricas=R["maior_dif_metricas"], maior_dif_ri=R["maior_dif_ri"], ri=R["ri"],
                       desempenho=R["met_re"].groupby("variante").mean(numeric_only=True).drop(columns=["seed"]).round(4).to_dict("index"),
                       referencias=R["refs"], ap_empates=R["ap"], minutos_do_notebook=R["minutos"], fases_em_minutos=R["fases"],
                       erro_por_faixa=em_json(R["faixas"]) if "faixas" in R else None, erros_grandes=R.get("grandes"), latentes_da_kan=R.get("latentes"))
    (saida / "resumo.json").write_text(json.dumps(resumo_json, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n".join(texto))
    print(f"\nGravado em {saida}/")


if __name__ == "__main__":
    main()
