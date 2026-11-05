#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparar_completo_vs_rapido.py — por que o notebook RÁPIDO (sem folds) parece pior que o COMPLETO (o `kanscnn (4).ipynb`)?
===========================================================================================================================

Pergunta do usuário (2026-11-02): «por que deu diferença do `kanscnn (4).ipynb` (a 2ª execução, notebook completo) para o notebook rápido que está
agora? O que fez o resultado piorar?». Em números, o que parece pior é o ΔVmax e a intensificação rápida (RI); o Vmax não piorou. Os dois experimentos
NÃO medem a mesma coisa (o completo testa todos os ciclones em 5 folds e mostra a média de 25 treinos por fold; o rápido treina em 2003–2014 e testa só
2017, com 1 treino), então comparar os números das tabelas é comparar testes diferentes. Este script separa o que é diferença de TESTE do que é
diferença de MODELO/PROTOCOLO, comparando as duas execuções nas MESMAS janelas.

O QUE FAZ
---------
 1. Confere o código: o notebook baixado do Kaggle × o do projeto (célula a célula, pela árvore sintática; o baixado só é lido como JSON, nunca
    executado) e o completo × o rápido (`verificar_equivalencia_rapido.py`).
 2. Reconstrói as 65.670 janelas (precisa dos 3 TCIR completos) e VALIDA o mapeamento de `idx_teste` nas 100 execuções do completo e na do rápido
    (rótulos idênticos; o teste do rápido é exatamente o conjunto das janelas de 2017). Sem isso nada abaixo vale e o script aborta.
 3. Nas 3.967 janelas de 2017 (o teste do rápido) compara: os modelos da validação cruzada (cada seed, a média e o ensemble das 5), o rápido (1 seed)
    e o baseline de histórico nos DOIS protocolos (ajustado nos outros folds × ajustado em 2003–2014).
 4. Decompõe a diferença «headline do completo → rápido» em efeito do TESTE (mesmos modelos, só trocando as janelas) e efeito do MODELO/PROTOCOLO
    (nas mesmas janelas), com a variação entre seeds como régua de ruído e bootstrap por ciclone (só os ciclones de 2017).
 5. Mede o efeito dos empates no AP do baseline de histórico (item 18 da seção 12 do CLAUDE.md).
 6. Traz a evidência local do ajuste de hiperparâmetros (peso de ΔVmax 0,5 → 0,2) para dizer quanto dela pode explicar.

USO
---
    python3 -I scripts/analise/comparar_completo_vs_rapido.py --dados <pasta com os 3 TCIR-*.h5 completos> \\
        [--notebook-baixado "/caminho/kanscnn (4).ipynb"] [--completo resultados_pipeline/saida_pipeline] \\
        [--rapido resultados_rapido_base_seed0_sem_folds/base_seed0] [--saida resultados_comparacao_completo_rapido]

`-I` (modo isolado): o notebook e os zips baixados são dados não confiáveis; só se lêem JSON, CSV e `.npz` sem pickle.

SAÍDAS (em ``resultados_comparacao_completo_rapido/``)
------------------------------------------------------
    relatorio.md     o relatório legível, gerado a partir dos números
    resumo.json      todos os números (para conferir/citar)
    verificacoes.csv as verificações de integridade (✔/✖)
    tabelas/*.csv    uma tabela por análise
    figuras/*.png    a decomposição e o ΔVmax por ano
    verificacao_equivalencia_rapido.txt   a saída de `verificar_equivalencia_rapido.py`
"""
import argparse
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps

RAIZ = Path(__file__).resolve().parents[2]  # raiz do projeto (este arquivo fica em scripts/analise/)
sys.path.insert(0, str(RAIZ / "scripts" / "dados"))  # com `python3 -I` a pasta do script não entra sozinha em sys.path
sys.path.insert(0, str(RAIZ / "scripts" / "analise"))
import estatisticas_janelas as ej           # noqa: E402  (mesmas regras de janelas do notebook)
import analisar_resultados_pipeline as arp  # noqa: E402  (métricas, bootstrap e baseline: as mesmas funções da análise da 2ª execução)

VARIANTE = "cnn_lstm_kan"                   # a proposta: a única que o rápido treinou
LIMIAR_RI = ej.LIMIAR_RI_NOS                # 30 nós
ANO_TESTE_RAPIDO = 2017                     # o teste do rápido (divisão por ano, como no TCC)
ANO_FIM_TREINO = 2014                       # o treino do rápido (2003–2014)
SEED_BOOT = ej.SEED_ESTRUTURAL              # a mesma semente do bootstrap do notebook (2024)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt(x, nd: int = 2) -> str:
    return arp.fmt(x, nd)


# ----------------------------------------------------------------------------- 1. código dos notebooks
def _instrucoes(arvore: ast.Module):
    return [ast.unparse(n) for n in arvore.body]


def _strings(no: ast.AST) -> set:
    return {c.value for c in ast.walk(no) if isinstance(c, ast.Constant) and isinstance(c.value, str)}


def comparar_codigo(baixado: Path, projeto: Path) -> dict:
    """Célula a célula, pela ÁRVORE SINTÁTICA (ignora comentários e espaços). Só JSON + `ast.parse`: nada do baixado é executado.
    Para cada célula de código diferente diz apenas o que mudou NO CÓDIGO (instruções só de um lado; para atribuições do mesmo nome, as
    cadeias de texto que entram/saem), sem reproduzir comentários nem o texto das células Markdown."""
    a = json.loads(baixado.read_text(encoding="utf-8"))["cells"]
    b = json.loads(projeto.read_text(encoding="utf-8"))["cells"]
    r = dict(celulas_baixado=len(a), celulas_projeto=len(b), mesmos_tipos=[x["cell_type"] for x in a] == [x["cell_type"] for x in b],
             celulas_codigo=sum(x["cell_type"] == "code" for x in a), codigo_diferente=[], texto_diferente=0)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = "".join(x["source"]), "".join(y["source"])
        if x["cell_type"] != "code":
            r["texto_diferente"] += int(sx != sy)
            continue
        if sx == sy:
            continue
        ta, tb = ast.parse(sx), ast.parse(sy)
        if ast.dump(ta) == ast.dump(tb):                         # só comentário/espaço
            continue
        ia, ib = _instrucoes(ta), _instrucoes(tb)
        so_a, so_b = [s for s in ia if s not in ib], [s for s in ib if s not in ia]
        det = dict(celula=i, instrucoes_so_no_baixado=len(so_a), instrucoes_so_no_projeto=len(so_b), mudancas=[])
        alvos_a = {n.targets[0].id: n for n in ta.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        alvos_b = {n.targets[0].id: n for n in tb.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        for nome in sorted(set(alvos_a) | set(alvos_b)):
            if nome in alvos_a and nome in alvos_b:
                if ast.dump(alvos_a[nome]) != ast.dump(alvos_b[nome]):
                    sa, sb = _strings(alvos_a[nome].value), _strings(alvos_b[nome].value)
                    det["mudancas"].append(dict(atribuicao=nome, textos_so_no_baixado=sorted(sa - sb), textos_so_no_projeto=sorted(sb - sa)))
            elif nome in alvos_a:
                det["mudancas"].append(dict(atribuicao=nome, so_no="baixado", instrucao=ast.unparse(alvos_a[nome])[:80]))
            else:
                det["mudancas"].append(dict(atribuicao=nome, so_no="projeto", instrucao=ast.unparse(alvos_b[nome])[:80]))
        r["codigo_diferente"].append(det)
    return r


def rodar_equivalencia(completo: Path, rapido: Path):
    p = subprocess.run([sys.executable, "-I", str(RAIZ / "scripts" / "verificacao" / "verificar_equivalencia_rapido.py"), "--completo", str(completo), "--rapido", str(rapido)],
                       capture_output=True, text=True, timeout=600)
    return p.returncode, p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")


# ----------------------------------------------------------------------------- 2. execuções
def carregar_rapido(dir_runs: Path) -> dict:
    arqs = sorted(dir_runs.glob("*.npz"))
    if len(arqs) != 1:
        raise SystemExit(f"ERRO: esperava 1 execução do rápido em {dir_runs}, achei {len(arqs)}")
    with np.load(arqs[0], allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        return dict(arquivo=arqs[0].name, meta=meta, idx=d["idx_teste"].astype(np.int64),
                    y_vmax=d["y_vmax"].astype(np.float64), pred_vmax=d["pred_vmax"].astype(np.float64),
                    y_dvmax=d["y_dvmax"].astype(np.float64), pred_dvmax=d["pred_dvmax"].astype(np.float64),
                    y_dvmax_val=d["y_dvmax_val"].astype(np.float64), pred_dvmax_val=d["pred_dvmax_val"].astype(np.float64))


def vetor_completo(idx: np.ndarray, valores: np.ndarray, n: int) -> np.ndarray:
    v = np.full(n, np.nan)
    v[idx] = valores
    return v


# ----------------------------------------------------------------------------- 3. métricas
def ap_com_empates(real: np.ndarray, escore: np.ndarray) -> float:
    """Precisão média (área sob a curva precisão × revocação, em degraus) somando só sobre limiares DISTINTOS: escores iguais entram juntos.
    Sem empates é igual à `precisao_media` dos notebooks; com empates, a dos notebooks depende da ordem das janelas (item 18 da seção 12)."""
    real, escore = np.asarray(real, bool), np.asarray(escore, float)
    n1 = int(real.sum())
    if n1 == 0:
        return float("nan")
    ordem = np.argsort(-escore, kind="stable")
    s, r = escore[ordem], real[ordem]
    tp, fp = np.cumsum(r), np.cumsum(~r)
    ultimo = np.flatnonzero(np.r_[s[1:] != s[:-1], True])          # fim de cada grupo de escores iguais
    prec, rev = tp[ultimo] / (tp[ultimo] + fp[ultimo]), tp[ultimo] / n1
    return float(np.sum((rev - np.r_[0.0, rev[:-1]]) * prec))


def metricas_janelas(yv: np.ndarray, yd: np.ndarray, pv: np.ndarray, pd_: np.ndarray, m: np.ndarray) -> dict:
    """Vmax e ΔVmax nas janelas de `m` (máscara); RI (ΔVmax ≥ 30 nós) nas que têm rótulo. `pv`/`pd_` podem ser NaN (baselines não têm Vmax)."""
    out = {}
    mv = arp.metricas_regressao(yv[m], pv[m])
    md = arp.metricas_regressao(yd[m], pd_[m])
    out.update(n_vmax=mv["n"], rmse_vmax=mv["rmse"], mae_vmax=mv["mae"], vies_vmax=mv["vies"],
               n_dvmax=md["n"], rmse_dvmax=md["rmse"], mae_dvmax=md["mae"], vies_dvmax=md["vies"])
    ok = m & ~np.isnan(yd) & ~np.isnan(pd_)
    real = yd[ok] >= LIMIAR_RI
    out.update(eventos_ri=int(real.sum()), prevalencia_ri=float(real.mean()) if ok.any() else float("nan"),
               auc_ri=arp.auc_roc(real, pd_[ok]) if ok.any() else float("nan"),
               ap_ri=ap_com_empates(real, pd_[ok]) if ok.any() else float("nan"),
               ap_ri_notebook=arp.precisao_media(real, pd_[ok]) if ok.any() and real.any() else float("nan"))
    prev = out["prevalencia_ri"]
    out["ap_sobre_prevalencia"] = out["ap_ri"] / prev if prev and np.isfinite(prev) else float("nan")
    return out


def media_dp(linhas: list, chaves: list) -> dict:
    df = pd.DataFrame(linhas)
    return {k: (float(df[k].mean()), float(df[k].std(ddof=1)) if len(df) > 1 else float("nan")) for k in chaves}


# ----------------------------------------------------------------------------- 4. baseline de histórico nos dois protocolos
def baseline_historico_treino(jan: pd.DataFrame, serie: pd.DataFrame, treino: np.ndarray):
    """Réplica do `baselines_dvmax` do notebook rápido: regressão polinomial de grau 2 em Vmax(t) e na tendência de 3, 6 e 12 h,
    ajustada nas janelas de `treino` (máscara) com histórico e rótulo; devolve a previsão de ΔVmax para TODAS as janelas e a máscara do ajuste."""
    X = arp.features_historico(jan, serie)
    y = jan["dvmax"].to_numpy(float)
    tr = treino & ~np.isnan(X).any(1) & ~np.isnan(y)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Phi = arp.polinomio_grau2((X - mu) / sd)
    w, *_ = np.linalg.lstsq(Phi[tr], y[tr], rcond=None)
    return Phi @ w, tr


# ----------------------------------------------------------------------------- 4b. RI com alarme por limiar (POD, FAR, CSI), somando acertos e alarmes
def contagens_ri(y: np.ndarray, p: np.ndarray, limiar_alarme: float):
    """(acertos TP, alarmes falsos FP, eventos perdidos FN) de «alarme se ΔVmax previsto ≥ limiar» contra o evento ΔVmax real ≥ 30 nós."""
    ok = ~(np.isnan(y) | np.isnan(p))
    real, alarme = y[ok] >= LIMIAR_RI, p[ok] >= limiar_alarme
    return int((real & alarme).sum()), int((~real & alarme).sum()), int((real & ~alarme).sum())


def pod_far_csi(tp: int, fp: int, fn: int) -> dict:
    div = lambda a, b: float(a / b) if b > 0 else float("nan")
    return dict(pod=div(tp, tp + fn), far=div(fp, tp + fp), csi=div(tp, tp + fn + fp), alarmes=tp + fp, eventos=tp + fn)


def ri_cv_por_seed(execs: list, variante: str, seed: int, ano: np.ndarray, ano_alvo=None) -> dict:
    """RI do completo restrito às janelas de `ano_alvo` (None = todas as janelas de teste de cada fold). Para cada fold (uma execução), o limiar de alarme
    é o ajustado na validação interna DAQUELA execução (como no notebook) e os acertos/alarmes são somados nos 5 folds; `por_fold` guarda o POD/FAR/CSI
    de cada fold (com `ano_alvo=None` é o que o notebook gravou em `metricas_por_execucao.csv`)."""
    tot = {"fixo": np.zeros(3, int), "ajustado": np.zeros(3, int)}
    limiares, por_fold = [], []
    for e in sorted((e for e in execs if e["variante"] == variante and e["seed"] == seed), key=lambda e: e["fold"]):
        sel = np.ones(len(e["idx_teste"]), bool) if ano_alvo is None else ano[e["idx_teste"]] == ano_alvo
        lim = arp.limiar_ri_ajustado(e["y_dvmax_val"], e["pred_dvmax_val"], LIMIAR_RI)
        c_fixo = contagens_ri(e["y_dvmax"][sel], e["pred_dvmax"][sel], LIMIAR_RI)
        c_aj = contagens_ri(e["y_dvmax"][sel], e["pred_dvmax"][sel], lim)
        tot["fixo"] += c_fixo
        tot["ajustado"] += c_aj
        limiares.append(lim)
        por_fold.append(dict(fold=e["fold"], limiar=lim, **{k: v for k, v in pod_far_csi(*c_aj).items() if k in ("pod", "far", "csi")}))
    return dict(fixo=pod_far_csi(*tot["fixo"]), ajustado=pod_far_csi(*tot["ajustado"]), limiar_medio=float(np.mean(limiares)), por_fold=por_fold)


def ri_baseline_cv(jan: pd.DataFrame, serie: pd.DataFrame, fold: np.ndarray, ano: np.ndarray, ano_alvo=None) -> dict:
    """Mesmo cálculo para o baseline de histórico no protocolo do completo: por fold, a regressão é ajustada nos outros folds e o limiar de alarme
    é o que maximiza o CSI no PRÓPRIO treino (o `y_dvmax_val` dos baselines do notebook é o treino); conta as janelas de `ano_alvo` (None = todas) naquele fold."""
    X = arp.features_historico(jan, serie)
    y = jan["dvmax"].to_numpy(float)
    x_ok = ~np.isnan(X).any(1)
    tot = {"fixo": np.zeros(3, int), "ajustado": np.zeros(3, int)}
    limiares, por_fold = [], []
    for f in range(ej.N_FOLDS):
        tr = x_ok & ~np.isnan(y) & (fold != f)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Phi = arp.polinomio_grau2((X - mu) / sd)
        w, *_ = np.linalg.lstsq(Phi[tr], y[tr], rcond=None)
        prev = Phi @ w
        lim = arp.limiar_ri_ajustado(y[tr], prev[tr], LIMIAR_RI)
        te = (fold == f) if ano_alvo is None else (fold == f) & (ano == ano_alvo)
        c_fixo, c_aj = contagens_ri(y[te], prev[te], LIMIAR_RI), contagens_ri(y[te], prev[te], lim)
        tot["fixo"] += c_fixo
        tot["ajustado"] += c_aj
        limiares.append(lim)
        por_fold.append(dict(fold=f, limiar=lim, **{k: v for k, v in pod_far_csi(*c_aj).items() if k in ("pod", "far", "csi")}))
    return dict(fixo=pod_far_csi(*tot["fixo"]), ajustado=pod_far_csi(*tot["ajustado"]), limiar_medio=float(np.mean(limiares)), por_fold=por_fold)


def dif_max(a, b) -> float:
    """Maior diferença absoluta entre dois vetores; NaN contra NaN conta como igual, NaN contra número como diferença infinita."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ambos_nan = np.isnan(a) & np.isnan(b)
    um_nan = np.isnan(a) ^ np.isnan(b)
    if um_nan.any():
        return float("inf")
    return float(np.max(np.abs(np.where(ambos_nan, 0.0, a - b)))) if a.size else 0.0


# ----------------------------------------------------------------------------- 6. figuras
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#898781"      # completo (validação cruzada) · rápido · baseline sem imagens (cinza de de-ênfase: contexto)
CINZA_CLARO = "#c3c2b7"
SUPERFICIE, TINTA, TINTA_2, GRADE, EIXO = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def _milhar(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def _sinal(x: float, nd: int) -> str:
    """Número com sinal explícito e o sinal de menos tipográfico (o mesmo dos eixos)."""
    return f"{x:+.{nd}f}".replace("-", "−")


def estilo() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9.5,
                         "axes.facecolor": SUPERFICIE, "figure.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
                         "text.color": TINTA, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
                         "axes.edgecolor": EIXO, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRADE, "grid.linewidth": 0.8, "axes.axisbelow": True})


def _rotulo(ax, x, y, texto, dx=9, dy=0, ha="left", cor=TINTA, negrito=False):
    """Valor escrito ao lado do marcador, em tinta de texto (nunca na cor da série), com um halo da cor da superfície para não brigar com a grade."""
    import matplotlib.patheffects as pe
    ax.annotate(texto, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", color=cor, fontsize=9.5,
                fontweight="bold" if negrito else "normal", path_effects=[pe.withStroke(linewidth=3, foreground=SUPERFICIE)], zorder=6)


def figura_decomposicao(D: dict, caminho: Path) -> None:
    """Quatro painéis (cada um na sua escala; nunca um eixo duplo): Vmax, ΔVmax, AUC e AP de RI. Em cada um, da esquerda para a direita:
    os modelos do completo em TODAS as janelas → os MESMOS modelos só nas janelas de 2017 (o efeito do teste, em azul) → o rápido em 2017 (o efeito do
    notebook, em laranja). Pontos claros = as 5 seeds; marcador cheio = média; círculo vazio = ensemble; losango cinza = baseline sem imagens.
    Os números de cada coluna ficam embaixo (média; no rápido, o valor) e a mudança de cada trecho, escrita sobre o trecho."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation
    estilo()
    paineis = [("rmse_vmax", "RMSE de Vmax (nós)", "menor é melhor", False), ("rmse_dvmax", "RMSE de ΔVmax (nós)", "menor é melhor", True),
               ("auc_ri", "AUC da intensificação rápida (RI)", "maior é melhor", True), ("ap_ri", "AP da RI (precisão média)", "maior é melhor", True)]
    nomes_x = [f"modelos do completo\ntodas as janelas\n({_milhar(D['n_todas'])})", f"os mesmos modelos\nsó janelas de 2017\n({_milhar(D['n_2017'])})",
               f"notebook rápido\nsó janelas de 2017\n({_milhar(D['n_2017'])})"]
    desloc = np.linspace(-0.14, 0.14, 5)
    fig, eixos = plt.subplots(2, 2, figsize=(11.4, 8.9), dpi=150)
    for ax, (chave, titulo, sentido, com_base) in zip(eixos.ravel(), paineis):
        nd = 2 if chave.startswith("rmse") else 3
        v0 = np.array([x[chave] for x in D["seeds_todas"]])
        v1 = np.array([x[chave] for x in D["seeds_17"]])
        r = D["rapido"][chave]
        m0, m1 = float(v0.mean()), float(v1.mean())
        if com_base:
            b = [D["base_cv_todas"][chave], D["base_cv_17"][chave], D["base_rap_17"][chave]]
            ax.plot([0, 1, 2], b, color=CINZA, lw=1.0, zorder=2)
            ax.plot([0, 1, 2], b, ls="none", marker="D", ms=6, mfc=SUPERFICIE, mec=CINZA, mew=1.4, zorder=3)
        ax.scatter(0 + desloc, v0, s=24, color=AZUL, alpha=0.38, linewidths=0, zorder=3)
        ax.scatter(1 + desloc, v1, s=24, color=AZUL, alpha=0.38, linewidths=0, zorder=3)
        ax.plot([0, 1], [m0, m1], color=AZUL, lw=1.6, zorder=4, solid_capstyle="round")
        ax.plot([1, 2], [m1, r], color=LARANJA, lw=1.6, zorder=4, solid_capstyle="round")
        ax.plot([0, 1], [D["ens_todas"][chave], D["ens_17"][chave]], ls="none", marker="o", ms=7.5, mfc=SUPERFICIE, mec=AZUL, mew=1.5, zorder=5)
        ax.plot([0, 1], [m0, m1], ls="none", marker="o", ms=8.5, color=AZUL, mec=SUPERFICIE, mew=1.5, zorder=5)
        ax.plot([2], [r], ls="none", marker="o", ms=8.5, color=LARANJA, mec=SUPERFICIE, mew=1.5, zorder=5)
        ax.set_xlim(-0.55, 2.62)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(nomes_x, fontsize=8.6)
        ax.tick_params(axis="x", length=0, pad=6)
        # sob o nome de cada coluna: o número dela; e, sob os dois trechos, uma seta (na cor do trecho) com a mudança, para não escrever sobre as linhas
        def abaixo(pontos):      # transformação: x em dados, y a `pontos` pontos abaixo do eixo
            return ax.get_xaxis_transform() + ScaledTranslation(0, -pontos / 72.0, fig.dpi_scale_trans)
        for x_, txt in ((0, f"média {m0:.{nd}f}"), (1, f"média {m1:.{nd}f}"), (2, f"{r:.{nd}f}")):
            ax.text(x_, 0, txt, transform=abaixo(54), ha="center", va="top", fontsize=10, fontweight="bold" if x_ == 2 else "normal", color=TINTA, clip_on=False)
        for x0, x1, delta, cor, nome in ((0, 1, m1 - m0, AZUL, "teste"), (1, 2, r - m1, LARANJA, "notebook")):
            ax.annotate("", xy=(x1 - 0.1, 0), xytext=(x0 + 0.1, 0), xycoords=abaixo(88), textcoords=abaixo(88),
                        arrowprops=dict(arrowstyle="-|>", color=cor, lw=1.6, shrinkA=0, shrinkB=0, mutation_scale=9), annotation_clip=False)
            ax.text((x0 + x1) / 2, 0, f"{nome} {_sinal(delta, nd)}", transform=abaixo(84), ha="center", va="bottom", fontsize=9.5, color=TINTA, clip_on=False)
        ax.set_title(f"{titulo}  ·  {sentido}", loc="left", fontsize=10.5, fontweight="bold", color=TINTA, pad=10)
        ax.margins(y=0.16)
    leg = [Line2D([], [], ls="none", marker="o", ms=8, color=AZUL, mec=SUPERFICIE, label="notebook completo (validação cruzada): média das 5 seeds (pontos claros = cada seed)"),
           Line2D([], [], ls="none", marker="o", ms=7.5, mfc=SUPERFICIE, mec=AZUL, mew=1.5, label="ensemble das 5 seeds do completo"),
           Line2D([], [], ls="none", marker="o", ms=8, color=LARANJA, mec=SUPERFICIE, label="notebook rápido (1 treino, 1 seed)"),
           Line2D([], [], ls="none", marker="D", ms=6, mfc=SUPERFICIE, mec=CINZA, mew=1.4, label="baseline de histórico de Vmax (sem imagens), no protocolo de cada coluna")]
    fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0), handletextpad=0.5, columnspacing=2.0)
    fig.suptitle("A diferença entre os dois notebooks vem sobretudo do ano de teste (2017), não do modelo", x=0.012, y=0.992, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.012, 0.955, "Azul: mesmos modelos, só trocam as janelas (efeito do TESTE).  Laranja: nas mesmas janelas de 2017, do completo para o rápido (efeito do NOTEBOOK / protocolo).",
             ha="left", fontsize=9.5, color=TINTA_2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.945), h_pad=5.2, w_pad=2.0)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def figura_por_ano(df: pd.DataFrame, dv_rapido: float, dv_base_rapido: float, caminho: Path) -> None:
    """ΔVmax por ano de teste (modelos do completo, média das 5 seeds, cada ciclone previsto fora do treino) e o baseline de histórico; embaixo, a
    diferença rede − baseline por ano: a rede ganha em uns anos e perde em outros, e 2017 é um dos anos em que ela perde."""
    from matplotlib.lines import Line2D
    estilo()
    anos = df["ano"].to_numpy()
    gap = (df["rmse_dvmax"] - df["rmse_dvmax_historico"]).to_numpy()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.6, 7.8), dpi=150, sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1.0]))
    i17 = int(np.flatnonzero(anos == ANO_TESTE_RAPIDO)[0])
    a1.plot(anos, df["rmse_dvmax"], color=AZUL, lw=1.6, marker="o", ms=6, mec=SUPERFICIE, mew=1.3, zorder=4)
    a1.plot(anos, df["rmse_dvmax_historico"], color=CINZA, lw=1.4, marker="D", ms=5, mfc=SUPERFICIE, mec=CINZA, mew=1.3, zorder=3)
    a1.plot([ANO_TESTE_RAPIDO], [dv_rapido], ls="none", marker="o", ms=10, color=LARANJA, mec=SUPERFICIE, mew=1.5, zorder=5)        # o rápido, com a bolinha azul da rede por cima
    a1.plot([ANO_TESTE_RAPIDO], [df["rmse_dvmax"].iloc[i17]], ls="none", marker="o", ms=5, color=AZUL, mec=SUPERFICIE, mew=1.0, zorder=6)
    _rotulo(a1, ANO_TESTE_RAPIDO, df["rmse_dvmax"].iloc[i17], f"rede {df['rmse_dvmax'].iloc[i17]:.2f}", dx=-12, dy=9, ha="right")
    _rotulo(a1, ANO_TESTE_RAPIDO, dv_rapido, f"rápido {dv_rapido:.2f}", dx=12, dy=2, negrito=True)
    _rotulo(a1, ANO_TESTE_RAPIDO, df["rmse_dvmax_historico"].iloc[i17], f"baseline {df['rmse_dvmax_historico'].iloc[i17]:.2f}", dx=12, dy=-5, cor=TINTA_2)
    j = int(np.argmax(df["rmse_dvmax"].to_numpy()))
    if j != i17:
        _rotulo(a1, anos[j], df["rmse_dvmax"].iloc[j], f"{df['rmse_dvmax'].iloc[j]:.2f}", dx=0, dy=11, ha="center")
    a1.set_ylabel("RMSE de ΔVmax (nós)")
    a1.set_title("ΔVmax: erro por ano de teste (cada ciclone previsto por modelos que não o viram)", loc="left", fontsize=11, fontweight="bold", pad=10)
    a1.margins(y=0.12)
    a1.set_xlim(anos.min() - 0.7, anos.max() + 2.6)
    cores = [AZUL if a == ANO_TESTE_RAPIDO else CINZA_CLARO for a in anos]
    a2.bar(anos, gap, width=0.6, color=cores, zorder=3)
    a2.axhline(0, color=EIXO, lw=1.0, zorder=2)
    a2.plot([ANO_TESTE_RAPIDO], [dv_rapido - dv_base_rapido], ls="none", marker="o", ms=8.5, color=LARANJA, mec=SUPERFICIE, mew=1.5, zorder=5)
    _rotulo(a2, ANO_TESTE_RAPIDO, gap[i17], f"rede {_sinal(gap[i17], 2)}", dx=-12, dy=8, ha="right")
    _rotulo(a2, ANO_TESTE_RAPIDO, dv_rapido - dv_base_rapido, f"rápido {_sinal(dv_rapido - dv_base_rapido, 2)}", dx=12, dy=0, negrito=True)
    a2.set_ylabel("rede − baseline (nós)")
    a2.set_title("Diferença rede − baseline de histórico: acima de zero = a rede erra mais", loc="left", fontsize=11, fontweight="bold", pad=10)
    a2.set_xticks(anos)
    a2.tick_params(axis="x", labelsize=9, length=0, pad=5)
    a2.margins(y=0.2)
    leg = [Line2D([], [], ls="none", marker="o", ms=8, color=AZUL, mec=SUPERFICIE, label="rede do completo (média das 5 seeds)"),
           Line2D([], [], ls="none", marker="D", ms=6, mfc=SUPERFICIE, mec=CINZA, mew=1.4, label="baseline de histórico (sem imagens)"),
           Line2D([], [], ls="none", marker="o", ms=8, color=LARANJA, mec=SUPERFICIE, label="notebook rápido (só existe em 2017)")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1.0), h_pad=2.0)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- 7. relatório
def _limite_ruido(dp: float, n: int = 5) -> float:
    """Limite de 95% para a diferença entre UM treino novo e a média de `n` treinos da mesma família: t(n−1) · dp · √(1 + 1/n). Com só 5 seeds o dp é
    uma estimativa grosseira (por isso o t de Student e não ±2 dp), e a régua só mede o ruído de treino, não a diferença de protocolo."""
    return float(sps.t.ppf(0.975, n - 1) * dp * np.sqrt(1.0 + 1.0 / n)) if dp is not None and np.isfinite(dp) else float("nan")


def _veredito_ruido(dif: float, dp: float, nd: int = 2, n: int = 5) -> str:
    lim = _limite_ruido(dp, n)
    if not np.isfinite(lim):
        return "—"
    razao = abs(dif) / lim
    return f"{'fora do' if razao > 1.0 else ('no limite do' if razao >= 0.9 else 'dentro do')} ruído de 1 treino, limite ±{lim:.{nd}f}"


def _parece(completo: float, rapido: float, menor_melhor: bool) -> str:
    return "melhor" if (rapido < completo) == menor_melhor else "pior"


def escrever_relatorio(R: dict, T: dict, verif: list, caminho: Path) -> None:
    hc = R["headline_completo_25_execucoes"]
    mj, tod = R["mesmas_janelas_2017"], R["todas_as_janelas_cv"]
    mm, rap, ens = mj["completo_media_dp"], mj["rapido"], mj["completo_ensemble"]
    bcv, brap, zero = mj["baseline_cv"], mj["baseline_rapido"], mj["zero"]
    mmt = tod["media_dp"]
    ri, gap, tam = R["ri_limiar"], R["gap_por_ano"], R["tamanhos"]
    df_ano: pd.DataFrame = T["por_ano"]
    b1, b2, b3, b4, b5, b6, b_ri = T["b1"], T["b2"], T["b3"], T["b4"], T["b5"], T["b6"], T["b_ri"]
    n_total, n_teste, n_ciclones_teste = R["n_janelas"], tam["rapido"]["teste"], R["ciclones_teste"]
    seeds17 = mj["completo_por_seed"]
    d = lambda ch: rap[ch] - mm[ch]["media"]                       # rápido − média dos modelos do completo, nas mesmas janelas
    sd = lambda ch: mm[ch]["dp"]
    abaixo = lambda ch: rap[ch] < min(x[ch] for x in seeds17)
    acima = lambda ch: rap[ch] > max(x[ch] for x in seeds17)
    txt_abaixo = lambda ch, nd: (("menor que a de qualquer uma das 5 seeds" if d(ch) < 0 else "maior que a de qualquer uma das 5 seeds") if (abaixo(ch) or acima(ch)) else "dentro do intervalo das 5 seeds")\
                                + f"; {_veredito_ruido(d(ch), sd(ch), nd)}"
    linha_b1 = lambda tarefa, metrica: b1[(b1.tarefa == tarefa) & (b1.metrica == metrica) & (b1.comparacao == "rapido − completo_modelo_unico")].iloc[0]
    l_v, l_d = linha_b1("vmax", "rmse"), linha_b1("dvmax", "rmse")
    efeito_teste_dv = mm["rmse_dvmax"]["media"] - mmt["rmse_dvmax"]["media"]
    efeito_teste_v = mm["rmse_vmax"]["media"] - mmt["rmse_vmax"]["media"]
    csi_cv = [x["ajustado"]["csi"] for x in ri["completo_por_seed"]]
    csi_cv_m, csi_cv_sd = float(np.mean(csi_cv)), float(np.std(csi_cv, ddof=1))
    csi_rap = ri["rapido"]["ajustado"]["csi"]
    ri_rows = {r["escore"]: r for r in b_ri.to_dict("records")}
    s0 = ri_rows["completo_seed0"]                                  # dif_*_vs_ref = (rápido) − (escore da linha)
    inclui0 = lambda lo, hi: lo <= 0.0 <= hi
    d_auc, d_ap = d("auc_ri"), d("ap_ri")
    dentro_v = abs(d("rmse_vmax")) <= _limite_ruido(sd("rmse_vmax"))
    dentro_d = abs(d("rmse_dvmax")) <= _limite_ruido(sd("rmse_dvmax"))
    i17 = int(np.flatnonzero(df_ano["ano"].to_numpy() == ANO_TESTE_RAPIDO)[0])
    L: list = []
    P = L.append
    mil = lambda n_: f"{int(n_):,}".replace(",", ".")                  # 65670 → 65.670

    P("# Por que o notebook rápido parece pior que o completo (`kanscnn (4).ipynb`)?\n")
    P(f"Gerado por `comparar_completo_vs_rapido.py` em {time.strftime('%Y-%m-%d')}. Todos os números vêm dos arquivos das duas execuções "
      "(`runs/*.npz` e tabelas) e das janelas reconstruídas dos dados; as verificações de integridade estão na seção 8. "
      "«Completo» é a 2ª execução (o notebook `kanscnn (4).ipynb`: validação cruzada em 5 folds, 5 seeds); «rápido» é o notebook rápido sem folds "
      "(1 treino, divisão por ano, teste em 2017). Aqui só se compara a proposta (`cnn_lstm_kan`), a única variante que o rápido treinou. "
      "Erros em nós; RMSE e MAE menores são melhores, AUC, AP e CSI maiores são melhores.\n")

    # ------------------------------------------------------------------ 1
    P("## 1. Resposta curta\n")
    P("**O que parece ter piorado**, lendo as tabelas dos dois notebooks lado a lado:\n")
    t1 = pd.DataFrame([
        ["RMSE de Vmax (nós)", fmt(hc["rmse_vmax"]["media"]), fmt(rap["rmse_vmax"]), _parece(hc["rmse_vmax"]["media"], rap["rmse_vmax"], True)],
        ["RMSE de ΔVmax (nós)", fmt(hc["rmse_dvmax"]["media"]), fmt(rap["rmse_dvmax"]), _parece(hc["rmse_dvmax"]["media"], rap["rmse_dvmax"], True)],
        ["AUC da RI", fmt(hc["auc_ri"]["media"], 3), fmt(rap["auc_ri"], 3), _parece(hc["auc_ri"]["media"], rap["auc_ri"], False)],
        ["AP da RI", fmt(hc["ap_ri"]["media"], 3), fmt(rap["ap_ri"], 3), _parece(hc["ap_ri"]["media"], rap["ap_ri"], False)],
        ["CSI da RI (limiar ajustado na validação)", fmt(hc.get("csi_ri_aj", {}).get("media"), 3), fmt(csi_rap, 3),
         _parece(hc["csi_ri_aj"]["media"], csi_rap, False) if "csi_ri_aj" in hc else "—"]],
        columns=["métrica", "completo: média de 25 treinos, 1 modelo cada", "rápido: 1 treino, teste de 2017", "o rápido parece"])
    P(arp.tabela_md(t1))
    P("")
    P("**Por que essa comparação não vale.** As duas colunas não medem a mesma coisa: o completo é a média de 25 treinos (5 folds × 5 seeds), cada um testado "
      f"em um quinto dos ciclones de todos os anos (≈ {mil(tam['completo_fold0']['teste'])} janelas por fold; {mil(n_total)} no total); "
      f"o rápido é **1** treino testado **só em 2017** ({mil(n_teste)} janelas, {n_ciclones_teste} ciclones). Se 2017 for mais difícil (ou mais fácil) que a média dos anos, "
      "os dois notebooks dão números diferentes mesmo treinando o mesmo modelo (a menos de um hiperparâmetro: seção 2). A comparação justa é nas **mesmas janelas**: "
      f"os 25 treinos do completo guardaram a previsão de cada janela, então dá para medi-los só nas {mil(n_teste)} janelas de 2017 "
      "(cada ciclone de 2017 foi previsto por modelos que não o viram no treino):\n")
    lin = []
    for nome, ch, nd in (("RMSE de Vmax (nós)", "rmse_vmax", 2), ("RMSE de ΔVmax (nós)", "rmse_dvmax", 2), ("AUC da RI", "auc_ri", 3), ("AP da RI", "ap_ri", 3)):
        lin.append([nome, f"{mm[ch]['media']:.{nd}f} ± {mm[ch]['dp']:.{nd}f}", f"{rap[ch]:.{nd}f}", f"{d(ch):+.{nd}f}", _veredito_ruido(d(ch), sd(ch), nd)])
    lin.append(["CSI da RI (limiar ajustado)", f"{csi_cv_m:.3f} ± {csi_cv_sd:.3f}", f"{csi_rap:.3f}", f"{csi_rap - csi_cv_m:+.3f}", _veredito_ruido(csi_rap - csi_cv_m, csi_cv_sd, 3)])
    P(arp.tabela_md(pd.DataFrame(lin, columns=[f"nas mesmas {mil(n_teste)} janelas de 2017", "completo: média ± dp das 5 seeds", "rápido (1 seed)", "rápido − completo", "leitura (limite de 95% para 1 treino novo)"])))
    P("")
    P("**Em quatro pontos:**\n")
    pct = 100 * efeito_teste_dv / (rap["rmse_dvmax"] - mmt["rmse_dvmax"]["media"])
    P(f"1. **O Vmax não piorou.** Nas mesmas janelas o rápido dá RMSE {rap['rmse_vmax']:.2f} e os modelos do completo {mm['rmse_vmax']['media']:.2f} ± {mm['rmse_vmax']['dp']:.2f}: "
      f"diferença de {d('rmse_vmax'):+.2f} nó ({_veredito_ruido(d('rmse_vmax'), sd('rmse_vmax'), 2)}); o bootstrap por ciclone dá {l_v['dif']:+.2f} [{l_v['ic95_inf']:+.2f}; {l_v['ic95_sup']:+.2f}]. "
      f"O que faz o Vmax parecer **{'melhor' if rap['rmse_vmax'] < hc['rmse_vmax']['media'] else 'pior'}** no rápido ({hc['rmse_vmax']['media']:.2f} → {rap['rmse_vmax']:.2f}) é o teste: 2017 é um ano "
      f"{'mais fácil' if efeito_teste_v < 0 else 'mais difícil'} para o Vmax (os mesmos modelos do completo passam de {mmt['rmse_vmax']['media']:.2f} em todas as janelas "
      f"para {mm['rmse_vmax']['media']:.2f} só em 2017, {efeito_teste_v:+.2f} nó).")
    P(f"2. **O ΔVmax «piorou» porque o teste mudou, não o notebook.** Os mesmos modelos do completo passam de RMSE {mmt['rmse_dvmax']['media']:.2f} em todas as janelas para "
      f"**{mm['rmse_dvmax']['media']:.2f}** só em 2017 ({efeito_teste_dv:+.2f} nó só por trocar as janelas). O rápido dá **{rap['rmse_dvmax']:.2f}**: "
      f"{d('rmse_dvmax'):+.2f} nó em relação a eles ({_veredito_ruido(d('rmse_dvmax'), sd('rmse_dvmax'), 2)}); bootstrap por ciclone {l_d['dif']:+.2f} [{l_d['ic95_inf']:+.2f}; {l_d['ic95_sup']:+.2f}]. "
      f"Ou seja, ≈ {pct:.0f}% da alta de {mmt['rmse_dvmax']['media']:.2f} para {rap['rmse_dvmax']:.2f} vem do ano de teste.")
    ic_auc, ic_ap = inclui0(s0["dif_auc_ic_inf"], s0["dif_auc_ic_sup"]), inclui0(s0["dif_ap_ic_inf"], s0["dif_ap_ic_sup"])
    if ic_auc and ic_ap:
        fecho_ri = (f"os dois intervalos incluem zero: com {zero['eventos_ri']} eventos de RI em 2017 não dá para afirmar que o rápido é pior na RI, "
                    "e os dados não dizem de onde viria a diferença que sobra (peso de ΔVmax 0,2, divisão temporal, sorte da seed: seção 7).")
    else:
        fecho_ri = (f"ao menos um intervalo exclui zero, então a diferença na RI não é só ruído de ciclones ({zero['eventos_ri']} eventos em 2017), "
                    "mas os dados não separam as causas possíveis (peso de ΔVmax 0,2, divisão temporal, seed: seção 7).")
    cota = lambda ch: 100 * (mm[ch]["media"] - mmt[ch]["media"]) / (rap[ch] - mmt[ch]["media"])     # parte da queda (todas as janelas → rápido) que é do teste
    P(f"3. **Na RI, boa parte da queda também vem do teste, mas sobra uma diferença pequena.** Só pelo ano: AUC {mmt['auc_ri']['media']:.3f} → {mm['auc_ri']['media']:.3f} e AP "
      f"{mmt['ap_ri']['media']:.3f} → {mm['ap_ri']['media']:.3f} nos modelos do completo (≈ {cota('auc_ri'):.0f}% da queda da AUC e ≈ {cota('ap_ri'):.0f}% da do AP, de todas as janelas até o rápido). Do completo para o rápido, nas mesmas janelas, restam AUC {d_auc:+.3f} "
      f"({txt_abaixo('auc_ri', 3)}), AP {d_ap:+.3f} ({txt_abaixo('ap_ri', 3)}) e CSI {csi_rap - csi_cv_m:+.3f} "
      f"({_veredito_ruido(csi_rap - csi_cv_m, csi_cv_sd, 3)}). No bootstrap por ciclone, rápido − seed 0 do completo dá AUC {s0['dif_auc_vs_ref']:+.3f} "
      f"[{s0['dif_auc_ic_inf']:+.3f}; {s0['dif_auc_ic_sup']:+.3f}] e AP {s0['dif_ap_vs_ref']:+.3f} [{s0['dif_ap_ic_inf']:+.3f}; {s0['dif_ap_ic_sup']:+.3f}]: " + fecho_ri)
    P(f"4. **O baseline sem imagens também piora em 2017, e a diferença rede − baseline é parecida nos dois protocolos.** O baseline de histórico vai de RMSE de ΔVmax "
      f"{tod['baseline_cv']['rmse_dvmax']:.2f} (todas as janelas) para {bcv['rmse_dvmax']:.2f} (2017); no protocolo do rápido (ajustado em 2003–2014) dá {brap['rmse_dvmax']:.2f}. "
      f"A rede fica {mm['rmse_dvmax']['media'] - bcv['rmse_dvmax']:+.2f} nó do baseline em 2017 com os modelos do completo e {rap['rmse_dvmax'] - brap['rmse_dvmax']:+.2f} com o rápido (seção 6). "
      f"Em {gap['anos']} anos de teste, a rede ganha do baseline em {gap['anos_rede_melhor']} e perde em {gap['anos'] - gap['anos_rede_melhor']}; 2017 é o {gap['posto_2017_pior']}º pior ano "
      f"para ela (diferença média entre os anos {gap['media']:+.2f}, desvio {gap['dp']:.2f}): «a rede não supera o histórico» em 2017 é um achado de **um ano** e não derruba nem confirma o da 2ª execução.\n")
    P("> Regra de leitura: o rápido é **um** modelo; compare-o com a média dos modelos do completo (tabela acima), não com o ensemble das 5 seeds "
      f"({ens['rmse_vmax']:.2f} em Vmax e {ens['rmse_dvmax']:.2f} em ΔVmax nestas janelas, AUC {ens['auc_ri']:.3f}, AP {ens['ap_ri']:.3f}), que é outro modelo, 5× mais caro.\n")

    # ------------------------------------------------------------------ 2
    P("## 2. O que é diferente entre os dois experimentos\n")
    cod, eq = R.get("codigo_baixado_vs_projeto"), R["equivalencia"]
    if cod:
        por_celula = {x["celula"]: x for x in cod["codigo_diferente"]}
        partes = []
        for cel in sorted(por_celula):
            for m in por_celula[cel]["mudancas"]:
                if m["atribuicao"] == "MODO":
                    partes.append(f"a célula de configuração (célula {cel}) tem uma linha a mais que fixa `MODO = \"completo\"` logo depois de `MODO = \"teste\"` (no projeto fica só o modo teste)")
                elif m["atribuicao"] == "_CAMPOS_SO_DE_EXECUCAO":
                    campos = ", ".join(f"`{c}`" for c in m["textos_so_no_projeto"])
                    partes.append(f"a célula do protocolo (célula {cel}) não tem {len(m['textos_so_no_projeto'])} campos que o projeto tirou da assinatura das execuções ({campos}), "
                                  "que não mudam o resultado; é por isso que a 2ª execução tem a assinatura `e6afa7eb` e os notebooks atuais calculam outra")
                else:
                    partes.append(f"a célula {cel} difere em `{m['atribuicao']}`")
        P("**O código é o mesmo.** "
          f"O `kanscnn (4).ipynb` tem as mesmas {cod['celulas_projeto']} células do `pipeline_tc_kaggle.ipynb` do projeto e o código é idêntico (pela árvore sintática) em "
          f"{cod['celulas_codigo'] - len(por_celula)} das {cod['celulas_codigo']} células de código. "
          f"{'As ' + str(len(por_celula)) + ' que diferem' if len(por_celula) != 1 else 'A única que difere'}: " + "; ".join(partes) + ". "
          f"{cod['texto_diferente']} células de texto (Markdown) têm só redação diferente.\n")
    ident = f" ({eq['definicoes_identicas']} definições idênticas)" if eq.get("definicoes_identicas") else ""
    P(f"Entre o completo e o rápido, `verificar_equivalencia_rapido.py` (código de saída {eq['codigo_saida']}) confirma que o modelo, o pré-processamento, as métricas e o núcleo do treino "
      f"(normalização dos alvos, AdamW, agenda OneCycle, perda, laço de épocas e *early stopping*) são idênticos{ident}; o que muda é o protocolo e **um** hiperparâmetro:\n")
    ptxt = R["config_diferencas"]["pesos_tarefas"]
    t2 = pd.DataFrame([
        ["Código do modelo e do treino", "o do notebook completo", "idêntico (verificado)"],
        ["Divisão dos dados", "5 folds agrupados por ciclone, todos os anos", "por ano, como no TCC: treino 2003–2014, validação 2015–2016, teste 2017"],
        ["Janelas de treino (por treino)", f"{mil(tam['completo_fold0']['treino'])} (fold 0)", f"{mil(tam['rapido']['treino'])}"],
        ["Janelas da validação (*early stopping*)", f"{mil(tam['completo_fold0']['validacao'])} (10% dos ciclones do treino do fold)", f"{mil(tam['rapido']['validacao'])} (2015–2016)"],
        ["Janelas de teste", f"≈ {mil(tam['completo_fold0']['teste'])} por fold; {mil(n_total)} no total (todos os ciclones)", f"{mil(n_teste)} ({n_ciclones_teste} ciclones de 2017)"],
        ["Treinos da proposta / seeds", "25 / 5", "1 / 1"],
        ["Peso de ΔVmax na perda", f"{ptxt['completo']['dvmax']}", f"**{ptxt['rapido']['dvmax']}** (único hiperparâmetro ajustado; seção 7)"],
        ["Número de manchete", "média de 25 treinos (e, à parte, o ensemble das 5 seeds)", "o único treino"],
        ["Baseline de histórico", "ajustado nos outros 4 folds", "ajustado em 2003–2014"],
        ["Semente de cada treino", "100·seed + fold", "100·seed"]], columns=["aspecto", "completo (`kanscnn (4)`)", "rápido (sem folds)"])
    P(arp.tabela_md(t2))
    P("")
    P("Campos do `config.json` que diferem entre as duas execuções: " + ", ".join(f"`{k}`" for k in R["config_diferencas"]) +
      " (os de folds e de divisão, o plano de seeds e variantes, o diagnóstico de treino do rápido e o peso de ΔVmax).\n")

    # ------------------------------------------------------------------ 3
    P(f"## 3. Lado a lado, nas mesmas janelas (as {mil(n_teste)} de 2017)\n")
    P("Cada linha do completo é um dos 5 treinos da proposta (seeds 0–4) avaliado só nas janelas de 2017 (reunindo os 5 folds). A coluna do CSI usa, em cada fold, o limiar de alarme "
      "ajustado na validação interna daquele treino (como no notebook); no baseline, o ajustado no próprio treino.\n")
    cols3 = ["rmse_vmax", "mae_vmax", "rmse_dvmax", "mae_dvmax", "auc_ri", "ap_ri", "ap_sobre_prevalencia"]
    linhas3 = [[f"completo, seed {x['seed']}"] + [x[c] for c in cols3] + [ri["completo_por_seed"][k]["ajustado"]["csi"]] for k, x in enumerate(seeds17)]
    linhas3 += [["**completo: média das 5 seeds**"] + [mm[c]["media"] for c in cols3] + [csi_cv_m],
                ["completo: desvio entre seeds"] + [mm[c]["dp"] for c in cols3] + [csi_cv_sd],
                ["completo: ensemble das 5 seeds"] + [ens[c] for c in cols3] + [float("nan")],
                ["**rápido (seed 0)**"] + [rap[c] for c in cols3] + [csi_rap],
                ["baseline de histórico, protocolo do completo"] + [bcv[c] for c in cols3] + [ri["baseline_cv"]["ajustado"]["csi"]],
                ["baseline de histórico, protocolo do rápido"] + [brap[c] for c in cols3] + [ri["baseline_rapido"]["ajustado"]["csi"]],
                ["prever ΔVmax = 0"] + [zero[c] for c in cols3] + [0.0]]
    P(arp.tabela_md(pd.DataFrame(linhas3, columns=["modelo", "RMSE Vmax", "MAE Vmax", "RMSE ΔVmax", "MAE ΔVmax", "AUC RI", "AP RI", "AP ÷ prevalência", "CSI RI (limiar ajustado)"]), nd=3))
    P("")
    lim_cv = float(np.mean([x["limiar_medio"] for x in ri["completo_por_seed"]]))
    fx = lambda r_: f"{fmt(r_['fixo']['pod'], 2)} / {fmt(r_['fixo']['far'], 2)} / {fmt(r_['fixo']['csi'], 2)}"
    fx_cv = " / ".join(fmt(float(np.mean([x["fixo"][k] for x in ri["completo_por_seed"]])), 2) for k in ("pod", "far", "csi"))
    P(f"A prevalência de RI em 2017 é {zero['prevalencia_ri']:.3f} ({zero['eventos_ri']} eventos em {mil(zero['n_dvmax'])} janelas com rótulo de ΔVmax); o AP de «prever 0» (sem empates) é a prevalência. "
      f"Limiar de alarme ajustado (média dos folds e seeds do completo): {lim_cv:.1f} nós; no rápido: {ri['rapido']['limiar_medio']:.0f} nós; nos baselines: {ri['baseline_cv']['limiar_medio']:.1f} (protocolo do completo) "
      f"e {ri['baseline_rapido']['limiar_medio']:.0f} (protocolo do rápido). Com o limiar fixo de 30 nós (POD / FAR / CSI): rápido {fx(ri['rapido'])}; completo (média das seeds) {fx_cv}.\n")
    P("![Decomposição: do teste e do notebook](figuras/01_decomposicao.png)\n")

    # ------------------------------------------------------------------ 4
    P("## 4. Quanto é do teste e quanto é do modelo / protocolo\n")
    P("Para cada métrica, três medidas da **mesma proposta**: nos 25 treinos do completo (manchete), nos modelos do completo só nas janelas de 2017 e no rápido. "
      "O «efeito do teste» é a mudança de janelas com os modelos fixos; o «efeito do notebook» é a mudança de modelo nas mesmas janelas. "
      "(A manchete é a média das métricas de cada treino/fold; as colunas seguintes agrupam as janelas dos 5 folds de cada seed, então diferem alguns milésimos dela.)\n")
    linhas4 = []
    for nome, ch, nd in (("RMSE de Vmax (nós)", "rmse_vmax", 2), ("RMSE de ΔVmax (nós)", "rmse_dvmax", 2), ("AUC da RI", "auc_ri", 3), ("AP da RI", "ap_ri", 3)):
        todas_, a17, r_ = mmt[ch]["media"], mm[ch]["media"], rap[ch]
        linhas4.append([nome, f"{hc[ch]['media']:.{nd}f}", f"{todas_:.{nd}f}", f"{a17:.{nd}f}", f"{r_:.{nd}f}", f"{a17 - todas_:+.{nd}f}", f"{r_ - a17:+.{nd}f}", f"{mm[ch]['dp']:.{nd}f}"])
    P(arp.tabela_md(pd.DataFrame(linhas4, columns=["métrica", "manchete do completo (25 treinos)", "completo, todas as janelas", "completo, só 2017", "rápido, só 2017",
                                                    "efeito do teste", "efeito do notebook", "ruído (dp entre seeds, só 2017)"])))
    P("")
    hab = lambda r_, z_: 1.0 - r_ / z_
    h_red_t, h_red_17 = hab(mmt["rmse_dvmax"]["media"], tod["zero"]["rmse_dvmax"]), hab(mm["rmse_dvmax"]["media"], zero["rmse_dvmax"])
    h_base_t, h_base_17 = hab(tod["baseline_cv"]["rmse_dvmax"], tod["zero"]["rmse_dvmax"]), hab(bcv["rmse_dvmax"], zero["rmse_dvmax"])
    P(f"Outra forma de ver o ΔVmax: o ganho sobre «prever 0» (1 − RMSE ÷ RMSE de prever 0). Prever 0 dá {tod['zero']['rmse_dvmax']:.2f} em todas as janelas e {zero['rmse_dvmax']:.2f} em 2017 "
      f"(o ΔVmax de 2017 varia menos). O ganho da rede cai de {h_red_t:.3f} para {h_red_17:.3f} ({h_red_17 - h_red_t:+.3f}) e o do baseline de histórico, de {h_base_t:.3f} para {h_base_17:.3f} "
      f"({h_base_17 - h_base_t:+.3f}): em 2017 os dois previsores acertam relativamente menos, {'a rede mais que o baseline' if (h_red_17 - h_red_t) < (h_base_17 - h_base_t) else 'o baseline mais que a rede'}.\n")
    P("Bootstrap por ciclone (só os ciclones de 2017; 2000 réplicas; Holm dentro desta tabela); diferença = rápido − completo, **positiva = o rápido erra mais**:\n")
    t4b = b1[["comparacao", "tarefa", "metrica", "ciclones", "janelas", "dif", "ic95_inf", "ic95_sup", "p_boot", "p_holm"]].copy()
    t4b["comparacao"] = (t4b["comparacao"].str.replace("completo_modelo_unico", "completo (modelo único)").str.replace("completo_ensemble", "completo (ensemble)")
                         .str.replace("rapido", "rápido"))
    P(arp.tabela_md(t4b, nd=3))
    P("")
    P("Contra o **modelo único** (a comparação certa) nenhuma diferença se distingue de zero; contra o **ensemble** o rápido erra mais, como esperado (o ensemble tem menos variância e custa 5 treinos). "
      "O intervalo mede só a variação entre os ciclones de 2017; **não** inclui o ruído de seed do rápido (uma seed só).\n")

    # ------------------------------------------------------------------ 5
    P("## 5. 2017 é um ano difícil para o ΔVmax e a RI\n")
    P("Mesmos modelos do completo (média das 5 seeds; cada ciclone previsto por modelos que não o viram), por ano do ciclone:\n")
    t5 = df_ano[["ano", "ciclones", "janelas_com_dvmax", "eventos_ri", "rmse_dvmax", "rmse_dvmax_historico", "dif_rede_hist", "rmse_dvmax_zero", "auc_ri", "ap_ri"]].copy()
    t5.columns = ["ano", "ciclones", "janelas com ΔVmax", "eventos de RI", "RMSE ΔVmax rede", "RMSE ΔVmax baseline", "rede − baseline", "RMSE ΔVmax prever 0", "AUC RI rede", "AP RI rede"]
    P(arp.tabela_md(t5, nd=3))
    P("")
    posto = lambda n_, adj: f"o {adj}" if n_ == 1 else f"o {n_}º {adj}"
    P(f"- Em 2017 o RMSE de ΔVmax da rede é {df_ano['rmse_dvmax'].iloc[i17]:.2f}, {posto(int((df_ano['rmse_dvmax'] > df_ano['rmse_dvmax'].iloc[i17]).sum() + 1), 'maior')} dos {len(df_ano)} anos "
      f"(mediana dos anos: {df_ano['rmse_dvmax'].median():.2f}); o AP da rede é {df_ano['ap_ri'].iloc[i17]:.3f}, {posto(int((df_ano['ap_ri'] < df_ano['ap_ri'].iloc[i17]).sum() + 1), 'menor')} (mediana: {df_ano['ap_ri'].median():.3f}).")
    P(f"- A diferença rede − baseline muda de sinal conforme o ano: vai de {gap['minimo'][1]:+.2f} ({gap['minimo'][0]}) a {gap['maximo'][1]:+.2f} ({gap['maximo'][0]}), com média {gap['media']:+.2f} e desvio {gap['dp']:.2f}; "
      f"a rede ganha em {gap['anos_rede_melhor']} dos {gap['anos']} anos. 2017 ({df_ano['dif_rede_hist'].iloc[i17]:+.2f}) fica a {(df_ano['dif_rede_hist'].iloc[i17] - gap['media']) / gap['dp']:.1f} desvios da média dos anos: um ano desfavorável à rede, não um defeito do notebook.")
    P(f"- Cada ano tem {gap['ciclones_min']} a {gap['ciclones_max']} ciclones e {gap['eventos_ri_min']} a {gap['eventos_ri_max']} eventos de RI: o AP e a diferença com o baseline têm ruído de amostragem grande "
      f"(IC 95% do AP do rápido em 2017: [{ri_rows['rapido']['ap_ic_inf']:.2f}; {ri_rows['rapido']['ap_ic_sup']:.2f}]). **Um ano de teste não decide se a rede supera o histórico.**")
    P("- Por que 2017 é mais difícil **não foi investigado** (seria preciso olhar os ciclones; o que se vê é que o ΔVmax de 2017 varia menos e que tanto a rede quanto o baseline acertam relativamente menos).\n")
    P("![ΔVmax por ano](figuras/02_dvmax_por_ano.png)\n")

    # ------------------------------------------------------------------ 6
    P("## 6. Rede × baseline de histórico, nos dois protocolos\n")
    P(f"Diferença rede − baseline nas mesmas janelas de 2017 com rótulo de ΔVmax (bootstrap por ciclone, 2000 réplicas; **positiva = a rede erra mais**):\n")
    ren = lambda s: (s.str.replace("completo_modelo_unico", "completo (modelo único)").str.replace("completo_ensemble", "completo (ensemble)")
                     .str.replace("baseline_cv", "baseline (protocolo do completo)").str.replace("baseline_rapido", "baseline (protocolo do rápido)").str.replace("rapido", "rápido"))
    t6 = pd.concat([b2, b3, b4])
    t6 = t6[t6.tarefa == "dvmax"][["comparacao", "metrica", "ciclones", "janelas", "dif", "ic95_inf", "ic95_sup", "p_boot", "p_holm"]].copy()
    t6["comparacao"] = ren(t6["comparacao"])
    P(arp.tabela_md(t6, nd=3))
    P("")
    P("RI sem limiar (AUC e AP; bootstrap por ciclone, 1000 réplicas; Holm sobre todas as diferenças da tabela). As colunas «dif» são **rápido − o escore da linha** (negativa = o rápido tem AUC ou AP menor):\n")
    t6b = b_ri[["escore", "auc", "auc_ic_inf", "auc_ic_sup", "ap", "ap_ic_inf", "ap_ic_sup", "dif_auc_vs_ref", "p_auc_holm", "dif_ap_vs_ref", "p_ap_holm"]].copy()
    t6b["escore"] = t6b["escore"].replace({"rapido": "rápido", "completo_ensemble": "completo, ensemble", "completo_seed0": "completo, seed 0", "baseline_cv": "baseline (protocolo do completo)",
                                           "baseline_rapido": "baseline (protocolo do rápido)"})
    P(arp.tabela_md(t6b, nd=3))
    P("")
    u = b5[(b5.tarefa == "dvmax") & (b5.metrica == "rmse")].iloc[0]
    e_ = b6[(b6.tarefa == "dvmax") & (b6.metrica == "rmse")].iloc[0]
    l_cv = b3[(b3.tarefa == "dvmax") & (b3.metrica == "rmse")].iloc[0]
    l_rp = b2[(b2.tarefa == "dvmax") & (b2.metrica == "rmse")].iloc[0]
    ambos_zero = inclui0(l_cv["ic95_inf"], l_cv["ic95_sup"]) and inclui0(l_rp["ic95_inf"], l_rp["ic95_sup"])
    P(f"Nos dois protocolos a rede {'erra **mais** que o baseline' if (l_cv['dif'] > 0 and l_rp['dif'] > 0) else 'não erra mais que o baseline nos dois protocolos'} em ΔVmax nas janelas de 2017 ({l_cv['dif']:+.2f} nó de RMSE com os modelos do completo e {l_rp['dif']:+.2f} com o rápido), "
      f"{'com IC que inclui zero nos dois' if ambos_zero else 'com IC que exclui zero em ao menos um'}. "
      f"Em **todos os anos e ciclones** do completo (a mesma conta, {mil(int(u['ciclones']))} ciclones com ΔVmax) o modelo único ficava {u['dif']:+.2f} [{u['ic95_inf']:+.2f}; {u['ic95_sup']:+.2f}] do baseline e o ensemble de 5 seeds "
      f"{e_['dif']:+.2f} [{e_['ic95_inf']:+.2f}; {e_['ic95_sup']:+.2f}] (RMSE de ΔVmax). Logo o rápido **não** contradiz a 2ª execução: em 2017 a rede perde por mais que na média dos anos, nos dois protocolos, e em outros anos ganha.\n")

    # ------------------------------------------------------------------ 7
    P("## 7. O que estes dados não separam\n")
    P(f"- **Peso de ΔVmax 0,5 → 0,2 (rápido).** Nas mesmas janelas o RMSE de Vmax {'e o de ΔVmax do rápido ficam dentro' if (dentro_v and dentro_d) else 'e o de ΔVmax do rápido NÃO ficam ambos dentro'} do ruído esperado de 1 treino"
      + (", então o peso não deixou marca visível nesses dois" if (dentro_v and dentro_d) else "") + f"; a diferença que sobra é na RI (AUC {d_auc:+.3f}, AP {d_ap:+.3f}), "
      "compatível com o tronco compartilhado (encoder + LSTM) receber menos sinal da tarefa de ΔVmax — **mas também com a divisão temporal e com a seed**. Não há como separar com os dados atuais. "
      "A evidência local (ajuste de hiperparâmetros no Mac, dados parciais, validação 2015–2016, 1 seed por ponto) aponta para um efeito pequeno:\n")
    aj = pd.DataFrame(R["ajuste_local"])
    aj = aj[aj.candidato.isin(["padrao", "peso_dvmax=0.2"])][["candidato", "seed", "epocas_max", "melhor_epoca", "rmse_val_vmax", "rmse_val_dvmax"]].copy()
    aj["candidato"] = aj["candidato"].replace({"padrao": "peso 0,5 (padrão)", "peso_dvmax=0.2": "peso 0,2"})
    aj.columns = ["candidato", "seed", "épocas", "melhor época", "RMSE Vmax (validação)", "RMSE ΔVmax (validação)"]
    P(arp.tabela_md(aj, nd=3))
    P("")
    P("  No ajuste, o ΔVmax na validação melhora 0,09 nó com o peso 0,2 (10 épocas, contra a média das duas seeds do padrão, que entre si já diferem 0,09 nó) e piora 0,10 nó (25 épocas, sem repetir a seed): |efeito| ≲ 0,1 nó, do tamanho do ruído entre seeds. Esse ajuste **não mediu RI**.")
    P("- **Divisão por ciclone (5 folds) × divisão temporal (2003–2014 → 2017).** Os modelos do completo treinaram também com outros ciclones de 2015–2017; o rápido só treinou com ciclones de 2003–2014 (2015–2016 entram só na escolha da época). "
      "Se isso pesasse muito, Vmax e ΔVmax do rápido sairiam piores nas mesmas janelas"
      + (", e não saíram" if (dentro_v and dentro_d) else "") + "; para a RI pode ter pesado um pouco, mas é indistinguível de seed.")
    P(f"- **Uma seed só.** O ruído entre seeds do completo nestas janelas (dp {sd('rmse_vmax'):.2f} nó em Vmax, {sd('rmse_dvmax'):.2f} em ΔVmax, {sd('auc_ri'):.3f} de AUC, {sd('ap_ri'):.3f} de AP) é a régua: o limite de 95% para um treino novo contra a média das 5 seeds é t(4) · dp · √(1 + 1/5) ≈ 3,04 · dp. "
      "O ruído do rápido em si não foi medido (uma seed só), e com LSTM na GPU a mesma seed não reproduz o treino (CLAUDE.md, seção 9.11). Com 5 seeds, o próprio dp é uma estimativa grosseira.")
    P("- **Modelo único × ensemble.** A manchete do `resultado--.md` inclui o ensemble das 5 seeds; o rápido é um modelo só.\n")
    P("**O experimento que separa tudo isso** é rodar o rápido com `HIPERPARAMETROS = \"antigos\"` (peso 0,5) e com `\"ajustados\"` (peso 0,2), na mesma divisão por ano, com 3 a 5 seeds cada "
      "(`SEED = (0, 1, 2)`: ≈ 59 min de relógio por conjunto no Kaggle, estimado); o que se compara é a RI (AUC, AP, CSI), porque Vmax e ΔVmax já estão iguais.\n")

    # ------------------------------------------------------------------ 8
    P("## 8. Verificações\n")
    P(arp.tabela_md(pd.DataFrame([["✔" if ok else "✖", v, det] for v, ok, det in verif], columns=["", "verificação", "detalhe"])))
    P("")
    P("Empates no AP do baseline de histórico (item 18 da seção 12 do CLAUDE.md): "
      + "; ".join(f"{k.replace('_', ' ')}: {v['valores_distintos']} valores distintos em {v['janelas']} janelas, AP {v['ap_notebook']:.4f} (notebook) × {v['ap_sem_empates']:.4f} (sem empates)" for k, v in R["empates_ap_baseline"].items())
      + ". A diferença é da ordem de 1e-3 (ou menos): **não muda nenhuma conclusão**. As previsões das redes não têm empates. Já o AP de «prever 0» do notebook rápido (0,053 no CSV dele) é o defeito dos empates em estado puro "
      f"(o certo é a prevalência, {zero['prevalencia_ri']:.3f}).\n")

    # ------------------------------------------------------------------ 9
    P("## 9. Como reproduzir\n")
    P("```bash\npython3 -I scripts/analise/comparar_completo_vs_rapido.py --dados <pasta com os 3 TCIR-*.h5 completos> \\\n    --notebook-baixado \"/caminho/kanscnn (4).ipynb\" \\\n"
      "    --completo resultados_pipeline/saida_pipeline \\\n    --rapido resultados_rapido_base_seed0_sem_folds/base_seed0\n```\n")
    P("Precisa de `numpy`, `pandas`, `scipy`, `h5py` e `matplotlib`; lê os `.npz` sem pickle e o notebook baixado só como JSON. Os 3 arquivos `TCIR-*.h5` entram só para reconstruir a tabela de janelas "
      "(o `info`; as imagens não são lidas, o cache `resultados_janelas/frac_invalido_ir.csv` já tem a fração de IR inválido).\n")
    caminho.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- 5. principal
def main() -> None:
    ap = argparse.ArgumentParser(description="Compara o notebook completo (kanscnn (4)) com o rápido sem folds, nas mesmas janelas de 2017.")
    ap.add_argument("--dados", type=Path, default=RAIZ / "data", help="pasta com os 3 TCIR-*.h5 COMPLETOS (o -001 tem 47.381 imagens)")
    ap.add_argument("--cache", type=Path, default=RAIZ / "resultados_janelas" / "frac_invalido_ir.csv")
    ap.add_argument("--completo", type=Path, default=RAIZ / "resultados_pipeline" / "saida_pipeline")
    ap.add_argument("--rapido", type=Path, default=RAIZ / "resultados_rapido_base_seed0_sem_folds" / "base_seed0")
    ap.add_argument("--notebook-baixado", type=Path, default=None, help="o kanscnn (4).ipynb baixado do Kaggle (lido só como JSON)")
    ap.add_argument("--notebook-completo", type=Path, default=RAIZ / "pipeline_tc_kaggle.ipynb")
    ap.add_argument("--notebook-rapido", type=Path, default=RAIZ / "pipeline_tc_kaggle_rapido.ipynb")
    ap.add_argument("--ajuste", type=Path, default=RAIZ / "resultados_ajuste" / "local_m1" / "ajuste.csv")
    ap.add_argument("--analise-anterior", type=Path, default=RAIZ / "resultados_analise_pipeline" / "tabelas",
                    help="tabelas da análise independente da 2ª execução (bootstrap_replicado.csv e bootstrap_ensemble.csv), para conferir o bootstrap em todas as janelas")
    ap.add_argument("--saida", type=Path, default=RAIZ / "resultados_comparacao_completo_rapido")
    ap.add_argument("--reps", type=int, default=2000, help="réplicas do bootstrap por ciclone (erros)")
    ap.add_argument("--reps-ri", type=int, default=1000, help="réplicas do bootstrap por ciclone (AUC e AP)")
    args = ap.parse_args()
    saida = args.saida
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)
    verif = []                                                  # (verificação, ok, detalhe)
    R = {}                                                      # tudo o que vai para o resumo.json

    # ---- 1. código
    log("1/7 código dos notebooks")
    if args.notebook_baixado is not None:
        cod = comparar_codigo(args.notebook_baixado, args.notebook_completo)
        R["codigo_baixado_vs_projeto"] = cod
        declaradas = {4, 32}                                    # `MODO` fixado em «completo» na célula de configuração; campos que saíram da assinatura
        achadas = {d["celula"] for d in cod["codigo_diferente"]}
        verif.append(("notebook baixado × notebook completo do projeto: mesmas 60 células e mesmo código, salvo as duas células declaradas",
                      cod["celulas_baixado"] == cod["celulas_projeto"] and cod["mesmos_tipos"] and achadas <= declaradas,
                      f"{cod['celulas_baixado']} células; código diferente nas células {sorted(achadas)}; texto Markdown diferente em {cod['texto_diferente']} células"))
    cod_rc, cod_txt = rodar_equivalencia(args.notebook_completo, args.notebook_rapido)
    (saida / "verificacao_equivalencia_rapido.txt").write_text(cod_txt, encoding="utf-8")
    verif.append(("completo × rápido: mesmo modelo, pré-processamento, métricas e núcleo do treino (`verificar_equivalencia_rapido.py`)", cod_rc == 0,
                  f"código de saída {cod_rc}; saída em verificacao_equivalencia_rapido.txt"))
    linha_peso = [l.strip() for l in cod_txt.splitlines() if "hiperparâmetros ajustados" in l]
    m_ident = re.search(r"definições idênticas \((\d+)\)", cod_txt)
    R["equivalencia"] = dict(codigo_saida=cod_rc, definicoes_identicas=int(m_ident.group(1)) if m_ident else None,
                             linha_hiperparametros=linha_peso[0] if linha_peso else None)

    # ---- 2. janelas e mapeamento
    log("2/7 janelas e mapeamento")
    jan, serie, fold, _ = arp.reconstruir_janelas(args.dados, args.cache)
    n = len(jan)
    execs_c = arp.carregar_execucoes(args.completo / "runs")
    r_map = arp.validar_mapeamento(execs_c, jan, fold)           # aborta se algo não bater
    verif.append(("janelas reconstruídas = as do completo (rótulos e fold das 100 execuções)", True,
                  f"{r_map['janelas']} janelas, {r_map['ciclones']} ciclones, {r_map['execucoes_conferidas']} execuções conferidas"))
    rap = carregar_rapido(args.rapido / "runs")
    ano = jan["ano"].to_numpy(int)
    yv, yd = jan["vmax"].to_numpy(np.float64), jan["dvmax"].to_numpy(np.float64)
    ids = jan["id_ciclone"].to_numpy()
    i17 = np.flatnonzero(ano == ANO_TESTE_RAPIDO)
    m17 = ano == ANO_TESTE_RAPIDO
    f32 = lambda a: a.astype(np.float32).astype(np.float64)      # os .npz guardam os rótulos em float32 (a mesma conversão de `validar_mapeamento`)
    ok_rot = (len(rap["idx"]) == len(np.unique(rap["idx"])) and np.array_equal(np.sort(rap["idx"]), i17)
              and np.array_equal(f32(yv[rap["idx"]]), rap["y_vmax"]) and np.array_equal(f32(yd[rap["idx"]]), rap["y_dvmax"], equal_nan=True))
    verif.append(("teste do rápido = exatamente as janelas de 2017, com os mesmos rótulos de Vmax e ΔVmax", bool(ok_rot),
                  f"{len(rap['idx'])} janelas (2017 tem {len(i17)}); {int((~np.isnan(yd[i17])).sum())} com rótulo de ΔVmax; {len(np.unique(ids[i17]))} ciclones"))
    if not ok_rot:
        raise SystemExit("ERRO: o teste do rápido não é o conjunto das janelas de 2017; nada abaixo vale.")
    R["n_janelas"] = int(n)
    R["ciclones_teste"] = int(len(np.unique(ids[i17])))
    seeds_c = sorted({e["seed"] for e in execs_c})
    variantes_c = sorted({e["variante"] for e in execs_c})
    P = arp.matrizes_previsao(execs_c, variantes_c, seeds_c, n)
    sem_cabeca_dvmax = [v for v in variantes_c if np.isnan(P[v]["dvmax"]).all()]          # a variante só-Vmax não tem cabeça de ΔVmax
    cobertura = all(not np.isnan(P[v]["vmax"]).any() for v in variantes_c) and all(not np.isnan(P[v]["dvmax"]).any() for v in variantes_c if v not in sem_cabeca_dvmax)
    verif.append(("completo: cada janela é prevista uma vez por (variante, seed), nos 5 seeds", bool(cobertura),
                  f"{len(variantes_c)} variantes × {len(seeds_c)} seeds × {n} janelas, sem lacunas (sem cabeça de ΔVmax: {', '.join(sem_cabeca_dvmax) or 'nenhuma'})"))
    pr_v, pr_d = vetor_completo(rap["idx"], rap["pred_vmax"], n), vetor_completo(rap["idx"], rap["pred_dvmax"], n)

    # métricas recalculadas × tabelas dos notebooks (completo: 25 execuções da proposta; rápido: a única)
    tab_c = pd.read_csv(args.completo / "tabelas" / "metricas_por_execucao.csv")
    dif_c = 0.0
    for e in execs_c:
        if e["variante"] != VARIANTE:
            continue
        lin = tab_c[(tab_c.variante == VARIANTE) & (tab_c.fold == e["fold"]) & (tab_c.seed == e["seed"])].iloc[0]
        mv, md = arp.metricas_regressao(e["y_vmax"], e["pred_vmax"]), arp.metricas_regressao(e["y_dvmax"], e["pred_dvmax"])
        dif_c = max(dif_c, abs(mv["rmse"] - lin["rmse_vmax"]), abs(md["rmse"] - lin["rmse_dvmax"]))
    verif.append(("completo: RMSE de Vmax e ΔVmax das 25 execuções recalculado das predições = `metricas_por_execucao.csv`", dif_c < 1e-4, f"diferença máx. {dif_c:.1e} nó"))
    tab_r = pd.read_csv(args.rapido / "tabelas" / "metricas_por_execucao.csv").iloc[0]
    m_rap_tudo = metricas_janelas(yv, yd, pr_v, pr_d, m17)
    dif_r = max(abs(m_rap_tudo["rmse_vmax"] - tab_r["rmse_vmax"]), abs(m_rap_tudo["mae_vmax"] - tab_r["mae_vmax"]),
                abs(m_rap_tudo["rmse_dvmax"] - tab_r["rmse_dvmax"]), abs(m_rap_tudo["mae_dvmax"] - tab_r["mae_dvmax"]),
                abs(m_rap_tudo["auc_ri"] - tab_r["auc_ri"]), abs(m_rap_tudo["ap_ri_notebook"] - tab_r["ap_ri"]))
    verif.append(("rápido: RMSE, MAE, AUC e AP recalculados das predições = `metricas_por_execucao.csv`", dif_r < 1e-4, f"diferença máx. {dif_r:.1e}"))

    # ---- 3. mesmas janelas (2017): modelos da validação cruzada, ensemble, rápido, baselines
    log("3/7 as mesmas 3.967 janelas de 2017")
    chaves = ["rmse_vmax", "mae_vmax", "vies_vmax", "rmse_dvmax", "mae_dvmax", "auc_ri", "ap_ri", "ap_sobre_prevalencia"]
    por_seed_17, por_seed_todas = [], []
    for k, s in enumerate(seeds_c):
        por_seed_17.append(dict(seed=s, **metricas_janelas(yv, yd, P[VARIANTE]["vmax"][k], P[VARIANTE]["dvmax"][k], m17)))
        por_seed_todas.append(dict(seed=s, **metricas_janelas(yv, yd, P[VARIANTE]["vmax"][k], P[VARIANTE]["dvmax"][k], np.ones(n, bool))))
    ens_v, ens_d = P[VARIANTE]["vmax"].mean(0), P[VARIANTE]["dvmax"].mean(0)
    m_ens_17 = metricas_janelas(yv, yd, ens_v, ens_d, m17)
    m_ens_todas = metricas_janelas(yv, yd, ens_v, ens_d, np.ones(n, bool))
    # baselines
    prev_cv, _ = arp.baseline_historico(jan, serie, fold)                         # fora da amostra, ajustado nos outros 4 folds (protocolo do completo)
    prev_rap, tr_rap = baseline_historico_treino(jan, serie, ano <= ANO_FIM_TREINO)   # ajustado em 2003–2014 (protocolo do rápido)
    nan_n = np.full(n, np.nan)
    m_base_cv_17 = metricas_janelas(yv, yd, nan_n, prev_cv, m17)
    m_base_cv_todas = metricas_janelas(yv, yd, nan_n, prev_cv, np.ones(n, bool))
    m_base_rap_17 = metricas_janelas(yv, yd, nan_n, prev_rap, m17)
    m_zero_17 = metricas_janelas(yv, yd, nan_n, np.zeros(n), m17)
    m_zero_todas = metricas_janelas(yv, yd, nan_n, np.zeros(n), np.ones(n, bool))
    # o baseline do rápido que o notebook gravou: confere a réplica
    tab_b = pd.read_csv(args.rapido / "tabelas" / "baselines_dvmax.csv").set_index("variante").loc["baseline_historico"]
    dif_b = max(abs(m_base_rap_17["rmse_dvmax"] - tab_b["rmse_dvmax"]), abs(m_base_rap_17["mae_dvmax"] - tab_b["mae_dvmax"]),
                abs(m_base_rap_17["auc_ri"] - tab_b["auc_ri"]), abs(m_base_rap_17["ap_ri_notebook"] - tab_b["ap_ri"]))
    verif.append(("baseline de histórico do rápido reproduzido aqui (regressão ajustada em 2003–2014) = `baselines_dvmax.csv` (RMSE, MAE, AUC, AP)", dif_b < 1e-3,
                  f"diferença máx. {dif_b:.1e}"))
    # a precisão média sem empates é igual à do notebook nas previsões das redes
    ap_ok = max(abs(x["ap_ri"] - x["ap_ri_notebook"]) for x in por_seed_17 + [m_ens_17, m_rap_tudo])
    verif.append(("AP das redes: a versão sem empates = a do notebook (as previsões das redes não têm empates)", ap_ok < 1e-12, f"diferença máx. {ap_ok:.1e}"))

    mm = media_dp(por_seed_17, chaves)
    mm_todas = media_dp(por_seed_todas, chaves)
    R["mesmas_janelas_2017"] = dict(
        completo_por_seed=por_seed_17, completo_media_dp={k: dict(media=v[0], dp=v[1]) for k, v in mm.items()}, completo_ensemble=m_ens_17,
        rapido=m_rap_tudo, baseline_cv=m_base_cv_17, baseline_rapido=m_base_rap_17, zero=m_zero_17)
    R["todas_as_janelas_cv"] = dict(por_seed=por_seed_todas, media_dp={k: dict(media=v[0], dp=v[1]) for k, v in mm_todas.items()},
                                    ensemble=m_ens_todas, baseline_cv=m_base_cv_todas, zero=m_zero_todas)

    # o «headline» do completo: a média das 25 execuções (uma por fold × seed) da tabela do notebook
    c25 = tab_c[tab_c.variante == VARIANTE]
    R["headline_completo_25_execucoes"] = {c: dict(media=float(c25[c].mean()), dp=float(c25[c].std(ddof=1)))
                                           for c in ("rmse_vmax", "mae_vmax", "rmse_dvmax", "mae_dvmax", "auc_ri", "ap_ri", "limiar_ri_aj", "pod_ri_aj", "far_ri_aj", "csi_ri_aj") if c in c25.columns}
    base_c = pd.read_csv(args.completo / "tabelas" / "baselines_dvmax_por_fold.csv")
    b_hist = base_c[base_c.variante == "baseline_historico"]
    R["headline_baseline_cv_5_folds"] = {c: float(b_hist[c].mean()) for c in ("rmse_dvmax", "mae_dvmax", "auc_ri", "ap_ri") if c in b_hist}

    # ---- 4. por ano (modelos da validação cruzada)
    log("4/7 por ano")
    linhas_ano = []
    for a in sorted(set(ano)):
        m = ano == a
        seeds_a = [metricas_janelas(yv, yd, P[VARIANTE]["vmax"][k], P[VARIANTE]["dvmax"][k], m) for k in range(len(seeds_c))]
        mb = metricas_janelas(yv, yd, nan_n, prev_cv, m)
        mz = metricas_janelas(yv, yd, nan_n, np.zeros(n), m)
        mean = lambda k: float(np.mean([x[k] for x in seeds_a]))
        linhas_ano.append(dict(ano=int(a), janelas=int(m.sum()), ciclones=int(len(np.unique(ids[m]))), janelas_com_dvmax=seeds_a[0]["n_dvmax"],
                               eventos_ri=seeds_a[0]["eventos_ri"], prevalencia_ri=seeds_a[0]["prevalencia_ri"], rmse_vmax=mean("rmse_vmax"), rmse_dvmax=mean("rmse_dvmax"),
                               rmse_dvmax_historico=mb["rmse_dvmax"], rmse_dvmax_zero=mz["rmse_dvmax"],
                               auc_ri=mean("auc_ri"), ap_ri=mean("ap_ri"), auc_ri_historico=mb["auc_ri"], ap_ri_historico=mb["ap_ri"]))
    df_ano = pd.DataFrame(linhas_ano)
    df_ano["dif_rede_hist"] = df_ano["rmse_dvmax"] - df_ano["rmse_dvmax_historico"]            # positiva = a rede erra mais que o baseline
    df_ano.to_csv(saida / "tabelas" / "por_ano_modelos_cv.csv", index=False)
    R["por_ano"] = df_ano.to_dict("records")
    g = df_ano["dif_rede_hist"].to_numpy()
    k17 = int(np.flatnonzero(df_ano["ano"].to_numpy() == ANO_TESTE_RAPIDO)[0])
    R["gap_por_ano"] = dict(
        media=float(g.mean()), dp=float(g.std(ddof=1)), anos=int(len(g)), anos_rede_melhor=int((g < 0).sum()),
        maximo=(int(df_ano["ano"].iloc[int(g.argmax())]), float(g.max())), minimo=(int(df_ano["ano"].iloc[int(g.argmin())]), float(g.min())),
        posto_2017_pior=int((g > g[k17]).sum() + 1), ciclones_min=int(df_ano["ciclones"].min()), ciclones_max=int(df_ano["ciclones"].max()),
        eventos_ri_min=int(df_ano["eventos_ri"].min()), eventos_ri_max=int(df_ano["eventos_ri"].max()))

    # ---- 5. bootstrap por ciclone nas janelas de 2017 (só os ciclones do teste, como no notebook rápido)
    log("5/7 bootstrap por ciclone (erros)")
    sub = i17
    y_v17, y_d17, ids17 = yv[sub], yd[sub], ids[sub]
    erros = {
        "rapido": {"vmax": arp.erros_medios(pr_v[sub][None, :], y_v17), "dvmax": arp.erros_medios(pr_d[sub][None, :], y_d17)},
        "completo_modelo_unico": {"vmax": arp.erros_medios(P[VARIANTE]["vmax"][:, sub], y_v17), "dvmax": arp.erros_medios(P[VARIANTE]["dvmax"][:, sub], y_d17)},
        "completo_ensemble": {"vmax": arp.erros_medios(ens_v[sub][None, :], y_v17), "dvmax": arp.erros_medios(ens_d[sub][None, :], y_d17)},
    }
    vazio = arp.erros_medios(np.full((1, len(sub)), np.nan), y_v17)
    erros["baseline_cv"] = {"vmax": vazio, "dvmax": arp.erros_medios(prev_cv[sub][None, :], y_d17)}
    erros["baseline_rapido"] = {"vmax": vazio, "dvmax": arp.erros_medios(prev_rap[sub][None, :], y_d17)}
    b1 = arp.bootstrap_por_ciclone(erros, "rapido", ["completo_modelo_unico", "completo_ensemble"], ids17, args.reps, SEED_BOOT)
    b2 = arp.bootstrap_por_ciclone(erros, "rapido", ["baseline_rapido"], ids17, args.reps, SEED_BOOT)
    b3 = arp.bootstrap_por_ciclone(erros, "completo_modelo_unico", ["baseline_cv"], ids17, args.reps, SEED_BOOT)
    b4 = arp.bootstrap_por_ciclone(erros, "completo_ensemble", ["baseline_cv"], ids17, args.reps, SEED_BOOT)
    for nome, d in (("rapido_vs_completo", b1), ("rede_vs_baseline_rapido", b2), ("rede_vs_baseline_cv_modelo_unico", b3), ("rede_vs_baseline_cv_ensemble", b4)):
        d.to_csv(saida / "tabelas" / f"bootstrap_{nome}.csv", index=False)
    R["bootstrap"] = {"rapido_vs_completo": b1.to_dict("records"), "rede_vs_baseline_rapido": b2.to_dict("records"),
                      "rede_vs_baseline_cv_modelo_unico": b3.to_dict("records"), "rede_vs_baseline_cv_ensemble": b4.to_dict("records")}

    # o mesmo bootstrap em TODAS as janelas (os 1373 ciclones): é o que a análise independente da 2ª execução já tinha; serve de conferência e de contexto
    erros_todas = {"completo_modelo_unico": {"vmax": arp.erros_medios(P[VARIANTE]["vmax"], yv), "dvmax": arp.erros_medios(P[VARIANTE]["dvmax"], yd)},
                   "completo_ensemble": {"vmax": arp.erros_medios(ens_v[None, :], yv), "dvmax": arp.erros_medios(ens_d[None, :], yd)},
                   "baseline_cv": {"vmax": arp.erros_medios(np.full((1, n), np.nan), yv), "dvmax": arp.erros_medios(prev_cv[None, :], yd)}}
    b5 = arp.bootstrap_por_ciclone(erros_todas, "completo_modelo_unico", ["baseline_cv"], ids, args.reps, SEED_BOOT)
    b6 = arp.bootstrap_por_ciclone(erros_todas, "completo_ensemble", ["baseline_cv"], ids, args.reps, SEED_BOOT)
    b5.to_csv(saida / "tabelas" / "bootstrap_rede_vs_baseline_cv_modelo_unico_todas_as_janelas.csv", index=False)
    b6.to_csv(saida / "tabelas" / "bootstrap_rede_vs_baseline_cv_ensemble_todas_as_janelas.csv", index=False)
    R["bootstrap"]["rede_vs_baseline_cv_modelo_unico_todas_as_janelas"] = b5.to_dict("records")
    R["bootstrap"]["rede_vs_baseline_cv_ensemble_todas_as_janelas"] = b6.to_dict("records")
    ant = args.analise_anterior
    if (ant / "bootstrap_replicado.csv").exists() and (ant / "bootstrap_ensemble.csv").exists():
        def _dif_boot(novo_, antigo_):
            a_ = antigo_[(antigo_.comparacao == "cnn_lstm_kan − baseline_historico") & (antigo_.tarefa == "dvmax")].set_index("metrica")
            n_ = novo_[novo_.tarefa == "dvmax"].set_index("metrica")
            return max(abs(a_.loc[m_, c_] - n_.loc[m_, c_]) for m_ in ("mae", "rmse") for c_ in ("dif", "ic95_inf", "ic95_sup"))
        dif_u = _dif_boot(b5, pd.read_csv(ant / "bootstrap_replicado.csv"))
        dif_e = _dif_boot(b6, pd.read_csv(ant / "bootstrap_ensemble.csv"))
        verif.append(("rede × baseline de histórico em todas as janelas (bootstrap por ciclone daqui) = o da análise independente da 2ª execução (modelo único e ensemble)",
                      max(dif_u, dif_e) < 1e-9, f"diferença máx. {max(dif_u, dif_e):.1e} (dif e IC de MAE e RMSE de ΔVmax)"))
    else:
        verif.append(("rede × baseline de histórico em todas as janelas = análise independente da 2ª execução", False, f"tabelas de comparação não encontradas em {ant}"))

    log("6/7 bootstrap por ciclone (AUC e AP de RI)")
    jan17 = jan.iloc[sub]
    escores = {"rapido": pr_d[sub], "completo_ensemble": ens_d[sub], "completo_seed0": P[VARIANTE]["dvmax"][0][sub],
               "baseline_cv": prev_cv[sub], "baseline_rapido": prev_rap[sub]}
    b_ri = arp.bootstrap_ri(jan17, escores, args.reps_ri, SEED_BOOT)
    b_ri.to_csv(saida / "tabelas" / "bootstrap_ri_2017.csv", index=False)
    R["bootstrap_ri"] = b_ri.to_dict("records")

    # ---- RI com alarme por limiar (POD, FAR, CSI): completo (por seed), rápido e baselines, nas janelas de 2017; e a conferência contra as tabelas dos notebooks
    ri_cv = [ri_cv_por_seed(execs_c, VARIANTE, s, ano, ANO_TESTE_RAPIDO) for s in seeds_c]
    lim_rap = arp.limiar_ri_ajustado(rap["y_dvmax_val"], rap["pred_dvmax_val"], LIMIAR_RI)
    ri_rap = dict(fixo=pod_far_csi(*contagens_ri(rap["y_dvmax"], rap["pred_dvmax"], LIMIAR_RI)),
                  ajustado=pod_far_csi(*contagens_ri(rap["y_dvmax"], rap["pred_dvmax"], lim_rap)), limiar_medio=lim_rap)
    ri_base_cv = ri_baseline_cv(jan, serie, fold, ano, ANO_TESTE_RAPIDO)
    lim_b = arp.limiar_ri_ajustado(yd[tr_rap], prev_rap[tr_rap], LIMIAR_RI)
    ri_base_rap = dict(fixo=pod_far_csi(*contagens_ri(yd[i17], prev_rap[i17], LIMIAR_RI)),
                       ajustado=pod_far_csi(*contagens_ri(yd[i17], prev_rap[i17], lim_b)), limiar_medio=lim_b)
    R["ri_limiar"] = dict(completo_por_seed=ri_cv, rapido=ri_rap, baseline_cv=ri_base_cv, baseline_rapido=ri_base_rap)
    # conferência: as mesmas contas, nos conjuntos de teste INTEIROS, reproduzem as tabelas dos notebooks (limiar ajustado, POD, FAR e CSI)
    dif_ri = 0.0
    for s in seeds_c:
        for pf in ri_cv_por_seed(execs_c, VARIANTE, s, ano, None)["por_fold"]:
            lin = tab_c[(tab_c.variante == VARIANTE) & (tab_c.fold == pf["fold"]) & (tab_c.seed == s)].iloc[0]
            dif_ri = max(dif_ri, dif_max([pf["limiar"], pf["pod"], pf["far"], pf["csi"]], [lin["limiar_ri_aj"], lin["pod_ri_aj"], lin["far_ri_aj"], lin["csi_ri_aj"]]))
    verif.append(("completo: limiar ajustado, POD, FAR e CSI de RI das 25 execuções, refeitos das previsões gravadas = `metricas_por_execucao.csv`", dif_ri < 1e-6, f"diferença máx. {dif_ri:.1e}"))
    ri_rap_tudo = pod_far_csi(*contagens_ri(rap["y_dvmax"], rap["pred_dvmax"], lim_rap))
    dif_ri_r = dif_max([lim_rap, ri_rap_tudo["pod"], ri_rap_tudo["far"], ri_rap_tudo["csi"]], [tab_r["limiar_ri_aj"], tab_r["pod_ri_aj"], tab_r["far_ri_aj"], tab_r["csi_ri_aj"]])
    verif.append(("rápido: limiar ajustado, POD, FAR e CSI de RI refeitos = `metricas_por_execucao.csv`", dif_ri_r < 1e-6, f"diferença máx. {dif_ri_r:.1e}"))
    base_por_fold = pd.read_csv(args.completo / "tabelas" / "baselines_dvmax_por_fold.csv")
    base_por_fold = base_por_fold[base_por_fold.variante == "baseline_historico"].set_index("fold")
    dif_ri_b = 0.0
    for pf in ri_baseline_cv(jan, serie, fold, ano, None)["por_fold"]:
        lin = base_por_fold.loc[pf["fold"]]
        dif_ri_b = max(dif_ri_b, dif_max([pf["limiar"], pf["pod"], pf["far"], pf["csi"]], [lin["limiar_ri_aj"], lin["pod_ri_aj"], lin["far_ri_aj"], lin["csi_ri_aj"]]))
    dif_ri_b2 = dif_max([lim_b, ri_base_rap["ajustado"]["pod"], ri_base_rap["ajustado"]["far"], ri_base_rap["ajustado"]["csi"]],
                        [tab_b["limiar_ri_aj"], tab_b["pod_ri_aj"], tab_b["far_ri_aj"], tab_b["csi_ri_aj"]])
    verif.append(("baseline de histórico: limiar ajustado, POD, FAR e CSI de RI refeitos (5 folds do completo e a divisão do rápido) = `baselines_dvmax*.csv`",
                  max(dif_ri_b, dif_ri_b2) < 1e-6, f"diferença máx. {max(dif_ri_b, dif_ri_b2):.1e} (completo: {dif_ri_b:.1e}; rápido: {dif_ri_b2:.1e})"))

    # ---- 6. empates no AP do baseline de histórico (item 18)
    log("7/7 empates no AP do baseline; evidência do ajuste")
    empates = {}
    for nome, prev, m in (("baseline_cv_todas_as_janelas", prev_cv, np.ones(n, bool)), ("baseline_cv_2017", prev_cv, m17), ("baseline_rapido_2017", prev_rap, m17)):
        ok = m & ~np.isnan(yd) & ~np.isnan(prev)
        real = yd[ok] >= LIMIAR_RI
        e = prev[ok]
        empates[nome] = dict(janelas=int(ok.sum()), valores_distintos=int(len(np.unique(e))), ap_notebook=arp.precisao_media(real, e), ap_sem_empates=ap_com_empates(real, e))
    R["empates_ap_baseline"] = empates

    # ---- evidência do ajuste local do peso de ΔVmax
    aj = pd.read_csv(args.ajuste)
    R["ajuste_local"] = aj[["etapa", "candidato", "seed", "epocas_max", "melhor_epoca", "rmse_val_vmax", "rmse_val_dvmax", "rmse_val_dvmax_min"]].to_dict("records")
    cfg_c = json.loads((args.completo / "config.json").read_text())
    cfg_r = json.loads((args.rapido / "config.json").read_text())
    dif_cfg = {k: dict(completo=cfg_c.get(k, "—"), rapido=cfg_r.get(k, "—")) for k in sorted(set(cfg_c) | set(cfg_r)) if cfg_c.get(k, "—") != cfg_r.get(k, "—")}
    R["config_diferencas"] = dif_cfg
    R["tamanhos"] = dict(completo_fold0=dict(treino=int(execs_c[0]["meta"]["n_treino"]), validacao=int(execs_c[0]["meta"]["n_val"]), teste=int(execs_c[0]["meta"]["n_teste"])),
                         rapido=dict(treino=int(rap["meta"]["n_treino"]), validacao=int(rap["meta"]["n_val"]), teste=int(rap["meta"]["n_teste"])))

    # ---- saídas
    D = dict(seeds_todas=por_seed_todas, seeds_17=por_seed_17, rapido=m_rap_tudo, ens_todas=m_ens_todas, ens_17=m_ens_17,
             base_cv_todas=m_base_cv_todas, base_cv_17=m_base_cv_17, base_rap_17=m_base_rap_17, n_todas=int(n), n_2017=int(len(i17)))
    T = dict(por_ano=df_ano, b1=b1, b2=b2, b3=b3, b4=b4, b5=b5, b6=b6, b_ri=b_ri)
    log("figuras")
    figura_decomposicao(D, saida / "figuras" / "01_decomposicao.png")
    figura_por_ano(df_ano, m_rap_tudo["rmse_dvmax"], m_base_rap_17["rmse_dvmax"], saida / "figuras" / "02_dvmax_por_ano.png")
    log("relatório")
    pd.DataFrame(verif, columns=["verificacao", "ok", "detalhe"]).to_csv(saida / "verificacoes.csv", index=False)
    cols = ["rmse_vmax", "mae_vmax", "vies_vmax", "rmse_dvmax", "mae_dvmax", "auc_ri", "ap_ri", "ap_sobre_prevalencia"]
    quadro = pd.DataFrame([dict(modelo=f"completo seed {x['seed']}", **{c: x[c] for c in cols}) for x in por_seed_17]
                          + [dict(modelo="completo MÉDIA 5 seeds", **{c: mm[c][0] for c in cols}),
                             dict(modelo="completo DP entre seeds", **{c: mm[c][1] for c in cols}),
                             dict(modelo="completo ENSEMBLE", **{c: m_ens_17[c] for c in cols}),
                             dict(modelo="RÁPIDO (seed 0)", **{c: m_rap_tudo[c] for c in cols}),
                             dict(modelo="baseline hist. (protocolo do completo)", **{c: m_base_cv_17[c] for c in cols}),
                             dict(modelo="baseline hist. (protocolo do rápido)", **{c: m_base_rap_17[c] for c in cols}),
                             dict(modelo="prever 0", **{c: m_zero_17[c] for c in cols})])
    quadro.to_csv(saida / "tabelas" / "mesmas_janelas_2017.csv", index=False)
    quadro2 = pd.DataFrame([dict(modelo="completo MÉDIA 5 seeds", **{c: mm_todas[c][0] for c in cols}), dict(modelo="completo DP entre seeds", **{c: mm_todas[c][1] for c in cols}),
                            dict(modelo="completo ENSEMBLE", **{c: m_ens_todas[c] for c in cols}), dict(modelo="baseline hist. (protocolo do completo)", **{c: m_base_cv_todas[c] for c in cols}),
                            dict(modelo="prever 0", **{c: m_zero_todas[c] for c in cols})])
    quadro2.to_csv(saida / "tabelas" / "todas_as_janelas_cv.csv", index=False)
    linhas_ri = [dict(modelo=f"completo seed {s}", **{f"{t}_{k}": v for t in ("fixo", "ajustado") for k, v in r_[t].items()}, limiar_medio=r_["limiar_medio"]) for s, r_ in zip(seeds_c, ri_cv)]
    linhas_ri += [dict(modelo=nm, **{f"{t}_{k}": v for t in ("fixo", "ajustado") for k, v in r_[t].items()}, limiar_medio=r_["limiar_medio"])
                  for nm, r_ in (("rápido", ri_rap), ("baseline (protocolo do completo)", ri_base_cv), ("baseline (protocolo do rápido)", ri_base_rap))]
    pd.DataFrame(linhas_ri).to_csv(saida / "tabelas" / "ri_por_limiar_2017.csv", index=False)
    pd.DataFrame(por_seed_17).to_csv(saida / "tabelas" / "completo_por_seed_2017.csv", index=False)
    R["tabelas_geradas"] = sorted(p.name for p in (saida / "tabelas").glob("*.csv"))
    escrever_relatorio(R, T, verif, saida / "relatorio.md")
    (saida / "resumo.json").write_text(json.dumps(arp.json_seguro(R), ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n==== VERIFICAÇÕES ====")
    for v, ok, det in verif:
        print(("  ✔ " if ok else "  ✖ ") + v + (f"  [{det}]" if det else ""))
    print("\n==== MESMAS JANELAS DE 2017 ====")
    print(quadro.round(3).to_string(index=False))
    print("\n==== TODAS AS JANELAS (CV) ====")
    print(quadro2.round(3).to_string(index=False))
    print("\n==== BOOTSTRAP: rápido − completo (positivo = o rápido erra mais) ====")
    print(b1[["comparacao", "tarefa", "metrica", "ciclones", "janelas", "dif", "ic95_inf", "ic95_sup", "p_boot", "p_holm"]].round(3).to_string(index=False))
    print("\n==== RI POR LIMIAR (2017) ====")
    print(pd.DataFrame(linhas_ri)[["modelo", "limiar_medio", "ajustado_pod", "ajustado_far", "ajustado_csi", "fixo_pod", "fixo_far", "fixo_csi"]].round(3).to_string(index=False))
    n_ruins = sum(1 for _, ok, _ in verif if not ok)
    print(f"\nSaída em {saida}  ({len(verif) - n_ruins} de {len(verif)} verificações ✔)")
    if n_ruins:
        sys.exit(1)


if __name__ == "__main__":
    main()
