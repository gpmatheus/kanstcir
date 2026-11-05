#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura da metodologia do notebook `pipeline_tc_kaggle.ipynb` (CNN encoder + LSTM + cabeças KAN multitarefa).

Só desenha o que o notebook fez. Os números escritos na figura NÃO são digitados à mão: vêm dos arquivos da execução
(`config.json`, `resumo_final.json`, `tabelas/metricas_por_execucao.csv`) e da análise independente
(`resultados_analise_pipeline/resumo_analise.json`); o nº de parâmetros de cada bloco é calculado das dimensões do `config.json`
e conferido contra o total gravado na execução (se não bater, o script aborta).

Uso (a partir da raiz do projeto):
    python3 -I scripts/relatorio/figura_metodologia.py
    python3 -I scripts/relatorio/figura_metodologia.py --saida docs/resultado_figuras --dpi 300

Saída: `<saida>/metodologia_pipeline.png` e `.pdf` (vetorial, para o artigo). As 20 miniaturas do tensor de entrada são recortadas de
`figuras/01_tensor_normal.png` da própria execução (dados reais); sem esse arquivo, o script desenha quadradinhos coloridos no lugar.
Código de saída 1 se algum texto passar da caixa que o contém (o aviso diz qual).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.image import imread  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]  # raiz do projeto (este arquivo fica em scripts/relatorio/)

# ----------------------------------------------------------------------------- aparência
plt.rcParams["font.family"] = "DejaVu Sans"        # a mesma família das figuras do notebook (e com Δ, σ, μ, ℝ, →, ≥)
plt.rcParams["pdf.fonttype"] = 42                   # texto editável no PDF
W, H = 120.0, 88.0                                  # área de desenho em "unidades"; a figura tem 12 × 8,8 pol: 1 unidade = 0,1 pol
FIG_W, FIG_H = 12.0, 8.8

TEXTO, TEXTO_2, PAINEL_FUNDO, PAINEL_BORDA, SETA = "#1F2933", "#52606D", "#F4F5F7", "#C5CBD3", "#4B5563"
C_DADOS, C_ENC, C_LSTM, C_KAN = "#4A5D73", "#2F6DB5", "#2E8B57", "#C9731B"
C_TREINO, C_AVAL, C_EXPL = "#6B4FA3", "#1F7F88", "#B0413E"
COR_VARIANTE = ["#1f5fa8", "#c9731b", "#3a8f5a", "#8a4fa8"]   # as cores das variantes nas figuras do notebook
FS_CORPO, DY = 7.9, 1.5                             # tamanho da fonte do corpo (pt) e distância entre linhas (unidades)


# ----------------------------------------------------------------------------- números (vêm dos arquivos da execução)
def contar_parametros(cfg: dict) -> dict:
    """Parâmetros de cada bloco, das dimensões do config.json (mesmas fórmulas das camadas do notebook)."""
    d, h = cfg["dim_embedding"], cfg["dim_oculta"]
    c_in, enc = 4, 0
    for c in cfg["canais_cnn"]:
        enc += c_in * c * 3 * 3 + 2 * c             # convolução 3×3 sem viés + BatchNorm (escala e deslocamento)
        c_in = c
    lado = cfg["recorte_final"] // 2 ** len(cfg["canais_cnn"])
    enc += c_in * lado * lado * d + d + 2 * d       # Linear (pesos + viés) + LayerNorm
    lstm = 4 * ((d + h) * h + 2 * h)                # LSTM de 1 camada (PyTorch: dois vieses)
    funcoes = cfg["kan_grid"] + cfg["kan_ordem"]    # funções-base de cada aresta
    dims = [h, *cfg["kan_ocultas"], 1]
    kan = sum(b * a * (1 + funcoes) for a, b in zip(dims[:-1], dims[1:]))   # peso_base + coeficientes da spline
    return dict(encoder=enc, lstm=lstm, kan=kan, funcoes=funcoes, total=enc + lstm + 2 * kan)


def carregar_numeros(saida_pipeline: Path, analise: Path) -> dict:
    cfg = json.loads((saida_pipeline / "config.json").read_text(encoding="utf-8"))
    res = json.loads((saida_pipeline / "resumo_final.json").read_text(encoding="utf-8"))
    ana = json.loads((analise / "resumo_analise.json").read_text(encoding="utf-8"))
    met = pd.read_csv(saida_pipeline / "tabelas" / "metricas_por_execucao.csv")
    par = contar_parametros(cfg)
    gravado = int(met.loc[met["variante"] == "cnn_lstm_kan", "parametros"].iloc[0])
    assert par["total"] == gravado, f"parâmetros calculados ({par['total']}) ≠ gravados na execução ({gravado})"
    lg, dd = ana["log"], res["dados"]
    n = dict(cfg=cfg, par=par, imagens=dd["imagens"], ids=dd["ciclones"] + dd["duplicatas"]["ids_unificados"],
             dup_removidas=dd["duplicatas"]["registros_removidos"], ids_unificados=dd["duplicatas"]["ids_unificados"],
             invalidos=dd["quadros_invalidos"], janelas=lg["janelas_log"]["janelas"], ciclones=lg["janelas_log"]["ciclones"],
             com_dvmax=lg["janelas_log"]["com_dvmax"], ri=lg["janelas_log"]["ri"], gpu=lg["gpu"]["nome"], horas=lg["tempo_treino"]["horas"],
             n_exec=res["execucoes"], n_var=len(cfg["variantes"]), n_folds=cfg["n_folds"], n_seeds=len(cfg["seeds"]), assinatura=res["assinatura"],
             parametros={v: int(g["parametros"].iloc[0]) for v, g in met.groupby("variante")})
    assert n["n_exec"] == n["n_folds"] * n["n_seeds"] * n["n_var"], "o plano não fecha: folds × seeds × variantes ≠ execuções"
    return n


def milhar(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def dec(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def sup(x) -> str:
    """128 → ¹²⁸ (expoente em Unicode: funciona em negrito e em qualquer tamanho)."""
    return str(x).translate(_SUP)


def exp10(x: float) -> str:
    """0.001 → '1e-3'."""
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e")


# ----------------------------------------------------------------------------- miniaturas reais do tensor de entrada
def _corridas(v: np.ndarray, minimo: int) -> list:
    out, ini = [], None
    for i, b in enumerate(v):
        if b and ini is None:
            ini = i
        elif not b and ini is not None:
            if i - ini >= minimo:
                out.append((ini, i))
            ini = None
    if ini is not None and len(v) - ini >= minimo:
        out.append((ini, len(v)))
    return out


def recortar_ladrilhos(png: Path | None, linhas: int = 4, colunas: int = 5):
    """Recorta os painéis (4 canais × 5 quadros) de `01_tensor_normal.png`. Devolve [linha][coluna] de arrays RGB, ou None."""
    if png is None or not png.exists():
        return None
    im = imread(png)[..., :3]
    nao_branco = (im < 0.96).any(axis=2)
    cols, lins = _corridas(nao_branco.mean(axis=0) > 0.45, 60), _corridas(nao_branco.mean(axis=1) > 0.45, 60)
    if len(cols) != colunas or len(lins) != linhas:
        print(f"[aviso] não achei a grade {linhas}×{colunas} em {png.name} (achei {len(lins)}×{len(cols)}): uso quadradinhos coloridos", file=sys.stderr)
        return None
    return [[im[a:b, c:d] for (c, d) in cols] for (a, b) in lins]


# ----------------------------------------------------------------------------- primitivas de desenho
class Medidor:
    """Mede a largura de um texto (em unidades de desenho) para quebrar linhas sem estourar a caixa."""

    def __init__(self, fig):
        self.r, self.dpi = fig.canvas.get_renderer(), fig.dpi

    def largura(self, s: str, fs: float, negrito: bool = False) -> float:
        fp = FontProperties(family="DejaVu Sans", size=fs, weight="bold" if negrito else "normal")
        w, _, _ = self.r.get_text_width_height_descent(s, fp, False)
        return w / (0.1 * self.dpi)


def quebrar(med: Medidor, texto: str, largura: float, fs: float) -> list:
    linhas, atual = [], ""
    for p in texto.split():
        cand = f"{atual} {p}".strip()
        if not atual or med.largura(cand, fs) <= largura:
            atual = cand
        else:
            linhas.append(atual)
            atual = p
    if atual:
        linhas.append(atual)
    return linhas


AVISOS: list = []


def corpo(ax, med, x, y, w, itens, fs=FS_CORPO, dy=DY, marcador="•", cor=TEXTO, limite_y=None, nome=""):
    """Escreve os itens (cada um quebrado para caber em `w`) a partir do topo `y`; devolve o y logo abaixo do último."""
    recuo = 1.25 if marcador else 0.0
    for it in itens:
        for i, ln in enumerate(quebrar(med, it, w - recuo, fs)):
            ax.text(x + recuo, y, ln, fontsize=fs, va="top", ha="left", color=cor, zorder=8)
            if i == 0 and marcador:
                ax.text(x, y, marcador, fontsize=fs, va="top", ha="left", color=cor, zorder=8)
            y -= dy
        y -= 0.35
    if limite_y is not None and y + 0.35 < limite_y - 0.05:
        AVISOS.append(f"texto passa da caixa '{nome}' em {limite_y - (y + 0.35):.2f} unidades")
    return y


def painel(ax, x, y, w, h, num, titulo, cor):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=PAINEL_FUNDO, ec=PAINEL_BORDA, lw=1.0, zorder=1))
    ax.add_patch(Circle((x + 2.7, y + h - 2.6), 1.5, fc=cor, ec="none", zorder=5))
    ax.text(x + 2.7, y + h - 2.6, str(num), color="white", fontsize=9.2, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(x + 5.0, y + h - 2.6, titulo, color=TEXTO, fontsize=10.2, fontweight="bold", ha="left", va="center", zorder=6)


def caixa(ax, x, y, w, h, titulo, cor, alt_cab=3.0, fs=8.5):
    """Caixa branca com faixa de título colorida; devolve o y onde começa o corpo."""
    base = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=cor, lw=1.2, zorder=3)
    ax.add_patch(base)
    cab = Rectangle((x, y + h - alt_cab), w, alt_cab, fc=cor, ec="none", zorder=4)
    ax.add_patch(cab)
    cab.set_clip_path(base)
    ax.text(x + w / 2, y + h - alt_cab / 2 - 0.05, titulo, color="white", fontsize=fs, fontweight="bold", ha="center", va="center", zorder=6)
    return y + h - alt_cab - 0.9


def seta(ax, p0, p1, cor=SETA, lw=1.5, ms=11, rad=0.0, estilo="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=estilo, mutation_scale=ms, lw=lw, color=cor, shrinkA=0, shrinkB=0, zorder=9,
                                 connectionstyle=f"arc3,rad={rad}"))


def spline_icone(ax, x, y, w, h, cor):
    """Curva suave (ilustração de uma função φ aprendida por aresta da KAN)."""
    t = np.linspace(0, 1, 80)
    f = 0.5 + 0.42 * np.tanh(3.0 * (t - 0.5)) + 0.09 * np.sin(8.5 * t)
    ax.plot([x, x + w], [y + h / 2, y + h / 2], color="#D5DAE1", lw=0.7, zorder=6)
    ax.plot(x + t * w, y + 0.5 + (h - 1.0) * f, color=cor, lw=1.6, zorder=7, solid_capstyle="round")


# ----------------------------------------------------------------------------- a figura
def desenhar(n: dict, png_tensor: Path | None, dpi: int):
    cfg, par = n["cfg"], n["par"]
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    med = Medidor(fig)

    # --- título
    ax.text(1.5, 85.9, "Metodologia do experimento", fontsize=16, fontweight="bold", color=TEXTO, ha="left", va="center")
    ax.text(1.5, 83.0, f"Estimativa da intensidade (Vmax) e da variação em {cfg['horizonte_horas']} h (ΔVmax) de ciclones tropicais a partir de sequências de "
            f"imagens de satélite (TCIR) · notebook pipeline_tc_kaggle.ipynb · Kaggle, {n['gpu']}, {dec(n['horas'], 2)} h, {n['n_exec']} treinos",
            fontsize=7.9, color=TEXTO_2, ha="left", va="center")

    # ============================================================ painel 1: dados e janelas
    p1 = dict(x=1.0, y=59.4, w=118.0, h=21.6)
    painel(ax, p1["x"], p1["y"], p1["w"], p1["h"], 1, "Dados e janelas (seções 2 e 3 do notebook)", C_DADOS)
    larg = [20.0, 29.0, 27.5, 31.5]
    gap = (p1["w"] - 4.0 - sum(larg)) / 3
    xs = [p1["x"] + 2.0 + sum(larg[:i]) + gap * i for i in range(4)]
    by, bh = p1["y"] + 1.2, 15.2
    lim = by + 0.3

    corpo(ax, med, xs[0] + 1.0, caixa(ax, xs[0], by, larg[0], bh, "TCIR", C_DADOS), larg[0] - 2.0, [
        "3 arquivos HDF5 · 2003–2017",
        f"{milhar(n['imagens'])} imagens · {milhar(n['ids'])} IDs",
        "201×201 px · 4 bandas",
        "rótulo: Vmax (best track)"], limite_y=lim, nome="TCIR")
    corpo(ax, med, xs[1] + 1.0, caixa(ax, xs[1], by, larg[1], bh, "Limpeza", C_DADOS), larg[1] - 2.0, [
        f"{n['dup_removidas']} duplicatas removidas; {n['ids_unificados']} IDs unificados",
        "bandas IR + PMW; recorte central 92×92 px",
        "NaN e valores > 1000 → 0 (como no TCC)",
        f"quadro com > {int(cfg['max_frac_nan_ir'] * 100)}% de IR ausente: inválido ({n['invalidos']})"], limite_y=lim, nome="Limpeza")
    corpo(ax, med, xs[2] + 1.0, caixa(ax, xs[2], by, larg[2], bh, "Janelas deslizantes", C_DADOS), larg[2] - 2.0, [
        f"{cfg['tam_janela'] + 2} quadros consecutivos a cada {cfg['passo_horas']} h ({cfg['tam_janela']} de entrada + 2 só para DTA/SDTA)",
        f"{milhar(n['janelas'])} janelas · {milhar(n['ciclones'])} ciclones",
        f"alvos: Vmax(t) e ΔVmax = Vmax(t+{cfg['horizonte_horas']} h) − Vmax(t); ΔVmax em {milhar(n['com_dvmax'])} janelas, RI em {dec(100 * n['ri'] / n['com_dvmax'])}%"],
        limite_y=lim, nome="Janelas")

    # caixa do tensor: miniaturas reais à esquerda, texto curto à direita
    corpo_y = caixa(ax, xs[3], by, larg[3], bh, f"Tensor por janela ({cfg['tam_janela']}, 4, {cfg['recorte_final']}, {cfg['recorte_final']})", C_DADOS, fs=8.3)
    tw, gx = 2.05, 0.26
    gx0, gy0 = xs[3] + 5.2, corpo_y - 1.35
    ladr = recortar_ladrilhos(png_tensor)
    cores_canal = ["#9AA0A6", "#4C8DC5", "#E0A030", "#C95A4F"]
    for r, canal in enumerate(["IR", "PMW", "DTA", "SDTA"]):
        y0 = gy0 - tw - r * (tw + gx)
        ax.text(gx0 - 0.5, y0 + tw / 2, canal, fontsize=6.3, ha="right", va="center", color=TEXTO_2, zorder=8)
        for c in range(cfg["tam_janela"]):
            x0 = gx0 + c * (tw + gx)
            if ladr is not None:
                ax.imshow(ladr[r][c], extent=(x0, x0 + tw, y0, y0 + tw), aspect="auto", zorder=6, interpolation="antialiased")
            else:
                ax.add_patch(Rectangle((x0, y0), tw, tw, fc=cores_canal[r], ec="none", zorder=6))
    for c, rot in enumerate(["t−12h", "t−9h", "t−6h", "t−3h", "t"]):
        ax.text(gx0 + c * (tw + gx) + tw / 2, gy0 + 0.3, rot, fontsize=5.0, ha="center", va="bottom", color=TEXTO_2, zorder=8)
    xt = gx0 + cfg["tam_janela"] * (tw + gx) + 0.7
    corpo(ax, med, xt, corpo_y - 0.2, xs[3] + larg[3] - xt - 0.4, [
        "z-score por canal (μ, σ só do treino)", "DTA e SDTA antes de girar",
        f"rotação aleatória, p = {dec(cfg['prob_rotacao'], 2)} (só no treino)", f"recorte {cfg['recorte_final']}×{cfg['recorte_final']}"],
        fs=6.9, dy=1.28, marcador="", limite_y=lim, nome="Tensor")
    for i in range(3):
        seta(ax, (xs[i] + larg[i] + 0.15, by + bh / 2), (xs[i + 1] - 0.15, by + bh / 2), ms=10)

    # ============================================================ painel 2: modelo
    p2 = dict(x=1.0, y=30.6, w=118.0, h=26.8)
    painel(ax, p2["x"], p2["y"], p2["w"], p2["h"], 2, "Modelo: CNN encoder → LSTM → cabeças KAN (seção 4)", C_ENC)
    larg2 = [13.0, 33.0, 30.0, 31.0]
    gap2 = (p2["w"] - 4.0 - sum(larg2)) / 3
    xm = [p2["x"] + 2.0 + sum(larg2[:i]) + gap2 * i for i in range(4)]
    my, mh = p2["y"] + 5.8, 16.4
    lim2 = my + 0.35

    # (a) entrada: 5 quadros em paralelo
    caixa(ax, xm[0], my, larg2[0], mh, "Entrada", "#6B7280")
    sq, pas = 1.6, 2.1
    y_sq0 = my + mh - 3.0 - 0.9
    ys_q = []
    for i, rot in enumerate(["t−12h", "t−9h", "t−6h", "t−3h", "t"]):
        y0 = y_sq0 - sq - i * pas
        ax.add_patch(Rectangle((xm[0] + 5.6, y0), sq, sq, fc="#E9ECEF", ec="#6B7280", lw=0.8, zorder=6))
        ax.text(xm[0] + 5.1, y0 + sq / 2, rot, fontsize=5.6, ha="right", va="center", color=TEXTO_2, zorder=8)
        ys_q.append(y0 + sq / 2)
    ax.text(xm[0] + larg2[0] / 2, my + 0.75, "4 canais × 64×64", fontsize=5.6, ha="center", va="center", color=TEXTO_2, zorder=8)

    # (b) encoder
    corpo(ax, med, xm[1] + 1.0, caixa(ax, xm[1], my, larg2[1], mh, "CNN encoder (pesos compartilhados)", C_ENC), larg2[1] - 2.0, [
        f"{len(cfg['canais_cnn'])} × [Conv 3×3 passo 2 + BatchNorm + ReLU], canais " + " → ".join(str(c) for c in cfg["canais_cnn"]) +
        f" ({cfg['recorte_final']} → {cfg['recorte_final'] // 2 ** len(cfg['canais_cnn'])} px)",
        f"Flatten → Dropout {dec(cfg['dropout'], 1)} → Linear ({cfg['canais_cnn'][-1] * (cfg['recorte_final'] // 2 ** len(cfg['canais_cnn'])) ** 2} → {cfg['dim_embedding']}) → LayerNorm",
        f"o MESMO encoder nos {cfg['tam_janela']} quadros: um vetor e(t) ∈ ℝ{sup(cfg['dim_embedding'])} por quadro",
        f"{milhar(par['encoder'])} parâmetros"], limite_y=lim2, nome="Encoder")
    for yq in ys_q:                                    # cinco setas paralelas, uma por quadro
        seta(ax, (xm[0] + larg2[0] + 0.1, yq), (xm[1] - 0.1, yq), ms=7, lw=1.0)

    # (c) LSTM
    corpo_y = caixa(ax, xm[2], my, larg2[2], mh, "LSTM", C_LSTM)
    cw, cg = 3.0, 2.55
    cx0 = xm[2] + (larg2[2] - (5 * cw + 4 * cg)) / 2
    cy0 = corpo_y - 0.4 - cw
    for i in range(5):
        x0 = cx0 + i * (cw + cg)
        ax.add_patch(FancyBboxPatch((x0, cy0), cw, cw, boxstyle="round,pad=0,rounding_size=0.6", fc="#E4F3EA", ec=C_LSTM, lw=1.0, zorder=6))
        ax.text(x0 + cw / 2, cy0 + cw / 2, "L", fontsize=6.5, fontweight="bold", color=C_LSTM, ha="center", va="center", zorder=8)
        ax.text(x0 + cw / 2, cy0 - 0.45, f"e{i + 1}", fontsize=5.8, ha="center", va="top", color=TEXTO_2, zorder=8)
        if i < 4:
            seta(ax, (x0 + cw + 0.1, cy0 + cw / 2), (x0 + cw + cg - 0.1, cy0 + cw / 2), ms=6, lw=0.9)
    corpo(ax, med, xm[2] + 1.0, cy0 - 2.6, larg2[2] - 2.0, [
        f"{cfg['dim_oculta']} unidades; lê e(t−12h) … e(t) em ordem",
        f"estado final h(t) ∈ (−1, 1){sup(cfg['dim_oculta'])}", f"{milhar(par['lstm'])} parâmetros"], limite_y=lim2, nome="LSTM")

    # (d) cabeças KAN
    corpo_y = caixa(ax, xm[3], my, larg2[3], mh, "Cabeças KAN (uma por tarefa)", C_KAN)
    sw = (larg2[3] - 2.0 - 0.8) / 2
    for i, rot in enumerate(["Vmax(t)", f"ΔVmax ({cfg['horizonte_horas']} h)"]):
        x0 = xm[3] + 1.0 + i * (sw + 0.8)
        ax.add_patch(FancyBboxPatch((x0, corpo_y - 5.6), sw, 5.4, boxstyle="round,pad=0,rounding_size=0.7", fc="#FCF3E8", ec=C_KAN, lw=0.9, zorder=6))
        ax.text(x0 + sw / 2, corpo_y - 1.0, rot, fontsize=7.4, fontweight="bold", color="#8A4A10", ha="center", va="center", zorder=8)
        spline_icone(ax, x0 + 1.3, corpo_y - 5.0, sw - 2.6, 3.0, C_KAN)
    k = cfg["kan_ocultas"]
    corpo(ax, med, xm[3] + 1.0, corpo_y - 6.2, larg2[3] - 2.0, [
        f"[{cfg['dim_oculta']} → {' → '.join(str(v) for v in k)} → 1]: B-splines de grau {cfg['kan_ordem']}, grid {cfg['kan_grid']} ({par['funcoes']} funções-base por aresta)",
        f"{milhar(par['kan'])} parâmetros por cabeça"], limite_y=lim2, nome="KAN")
    for i in (1, 2):                                   # encoder → LSTM → cabeças
        seta(ax, (xm[i] + larg2[i] + 0.1, my + mh / 2 - 1.5), (xm[i + 1] - 0.1, my + mh / 2 - 1.5), ms=10)

    # (e) variantes (ablações): mesmas cores das figuras do notebook
    ax.text(p2["x"] + 2.0, p2["y"] + 2.55, "Variantes\n(ablações)", fontsize=7.2, fontweight="bold", color=TEXTO, ha="left", va="center", linespacing=1.15, zorder=8)
    nomes = [("cnn_lstm_kan", "proposta: LSTM + KAN, Vmax e ΔVmax"), ("cnn_lstm_mlp", "sem KAN: cabeças MLP"),
             ("cnn_kan_1frame", "sem LSTM: só o quadro t"), ("cnn_lstm_kan_vmax", "sem multitarefa: só Vmax")]
    x_ch = p2["x"] + 13.0
    larg_ch = (p2["x"] + p2["w"] - 2.0 - x_ch - 3 * 1.0) / 4
    for i, (nome, desc) in enumerate(nomes):
        x0 = x_ch + i * (larg_ch + 1.0)
        ax.add_patch(FancyBboxPatch((x0, p2["y"] + 0.9), larg_ch, 3.4, boxstyle="round,pad=0,rounding_size=0.7", fc="white", ec=COR_VARIANTE[i], lw=1.0, zorder=3))
        ax.add_patch(Rectangle((x0, p2["y"] + 0.9), 0.9, 3.4, fc=COR_VARIANTE[i], ec="none", zorder=4))
        linha2 = f"{desc} · {milhar(n['parametros'][nome])} par."
        ax.text(x0 + 1.5, p2["y"] + 3.35, nome, fontsize=6.8, fontweight="bold", color=COR_VARIANTE[i], ha="left", va="center", zorder=8)
        ax.text(x0 + 1.5, p2["y"] + 1.95, linha2, fontsize=5.8, color=TEXTO_2, ha="left", va="center", zorder=8)
        if 1.5 + max(med.largura(linha2, 5.8), med.largura(nome, 6.8, True)) > larg_ch - 0.4:
            AVISOS.append(f"texto passa da caixa da variante '{nome}'")

    # ============================================================ linha de baixo: painéis 3, 4 e 5
    y3, h3 = 2.0, 26.6
    ytop = y3 + h3 - 4.9
    lim3 = y3 + 0.7

    # --- painel 3: treino e validação
    p3 = dict(x=1.0, w=44.0)
    painel(ax, p3["x"], y3, p3["w"], h3, 3, "Treino e validação (seções 5 e 6)", C_TREINO)
    nf = cfg["n_folds"]
    fw, fh, fg = 2.45, 1.5, 0.28
    fx0 = p3["x"] + 6.2
    for i in range(nf):
        y0 = ytop - 0.2 - fh - i * (fh + fg)
        ax.text(fx0 - 0.5, y0 + fh / 2, f"fold {i}", fontsize=5.8, ha="right", va="center", color=TEXTO_2, zorder=8)
        for j in range(nf):
            ax.add_patch(Rectangle((fx0 + j * (fw + fg), y0), fw, fh, fc=C_TREINO if i == j else "#D5DAE1", ec="none", zorder=6))
    yl = ytop - 0.2 - nf * (fh + fg) - 0.45
    ax.add_patch(Rectangle((fx0, yl - 1.0), 1.4, 1.0, fc=C_TREINO, ec="none", zorder=6))
    ax.text(fx0 + 1.9, yl - 0.5, "teste", fontsize=6.0, ha="left", va="center", color=TEXTO_2, zorder=8)
    ax.add_patch(Rectangle((fx0 + 6.0, yl - 1.0), 1.4, 1.0, fc="#D5DAE1", ec="none", zorder=6))
    ax.text(fx0 + 7.9, yl - 0.5, "treino", fontsize=6.0, ha="left", va="center", color=TEXTO_2, zorder=8)
    xr = fx0 + nf * (fw + fg) + 1.6
    corpo(ax, med, xr, ytop, p3["x"] + p3["w"] - xr - 1.2, [
        f"CV agrupada por ciclone (k = {nf}): os ciclones nunca se dividem entre treino e teste",
        f"{nf} folds × {n['n_seeds']} seeds × {n['n_var']} variantes = {n['n_exec']} treinos"], fs=7.3, dy=1.4, limite_y=yl - 1.0, nome="folds")
    corpo(ax, med, p3["x"] + 1.6, yl - 2.0, p3["w"] - 3.2, [
        f"validação interna: {int(cfg['frac_validacao_interna'] * 100)}% dos ciclones do treino → early stopping (paciência {cfg['paciencia']})",
        f"perda = {dec(cfg['pesos_tarefas']['vmax'], 1)}·MSE(Vmax) + {dec(cfg['pesos_tarefas']['dvmax'], 1)}·MSE(ΔVmax), este só nas janelas com rótulo em t+{cfg['horizonte_horas']} h",
        f"AdamW (lr {exp10(cfg['lr'])}, weight decay {exp10(cfg['weight_decay'])}) · OneCycle · ≤ {cfg['epocas_max']} épocas · lote {cfg['tam_lote']} · "
        "precisão mista (cabeças KAN em float32)"], fs=7.2, dy=1.38, limite_y=lim3, nome="treino")

    # --- painel 4: avaliação e estatística
    p4 = dict(x=47.0, w=35.0)
    painel(ax, p4["x"], y3, p4["w"], h3, 4, "Avaliação e estatística (seção 7)", C_AVAL)
    corpo(ax, med, p4["x"] + 1.6, ytop, p4["w"] - 3.2, [
        "RMSE, MAE e viés (nós) de Vmax e de ΔVmax, por execução",
        f"RI (ΔVmax ≥ {int(cfg['limiar_ri_nos'])} nós em {cfg['horizonte_horas']} h): POD, FAR e CSI com limiar {int(cfg['limiar_ri_nos'])} e com limiar ajustado na validação; AUC e AP",
        "baselines SEM imagens para ΔVmax: zero, média e histórico de Vmax (polinômio de grau 2)",
        f"teste principal: bootstrap por ciclone ({milhar(cfg['bootstrap_reps'])} réplicas) + Holm-Bonferroni",
        "apoio: 1 ponto por fold; Wilcoxon por janela e por execução (exploratórios)",
        f"diagnóstico: erro por faixa de intensidade; janelas com |erro| ≥ {int(cfg['outlier_erro_nos'])} nós"], fs=7.3, dy=1.4, limite_y=lim3, nome="avaliação")

    # --- painel 5: explicabilidade
    p5 = dict(x=84.0, w=35.0)
    painel(ax, p5["x"], y3, p5["w"], h3, 5, "Explicabilidade (seção 8)", C_EXPL)
    rng = np.random.default_rng(7)
    g = rng.normal(size=(26, 26))
    for _ in range(3):                                                   # suaviza o ruído (textura de nuvem ilustrativa)
        g = (g + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)) / 5
    yy, xx = np.mgrid[0:26, 0:26]
    calor = np.exp(-(((xx - 17) ** 2 + (yy - 8) ** 2) / 30.0))
    isz = 7.6
    ix, iy = p5["x"] + 2.0, ytop - 0.2 - isz
    ax.imshow(g, extent=(ix, ix + isz, iy, iy + isz), cmap="gray", aspect="auto", zorder=6)
    ax.imshow(calor, extent=(ix, ix + isz, iy, iy + isz), cmap="jet", alpha=0.55, aspect="auto", zorder=7)
    ax.text(ix + isz / 2, iy - 0.6, "Grad-CAM (esquema)", fontsize=6.0, fontweight="bold", color=C_EXPL, ha="center", va="top", zorder=8)
    sx = ix + isz + 3.4
    ax.add_patch(Rectangle((sx, iy), isz, isz, fc="white", ec="#C5CBD3", lw=0.8, zorder=5))
    spline_icone(ax, sx + 0.8, iy + 0.9, isz - 1.6, isz - 1.8, C_EXPL)
    ax.text(sx + isz / 2, iy - 0.6, "spline φ (esquema)", fontsize=6.0, fontweight="bold", color=C_EXPL, ha="center", va="top", zorder=8)
    corpo(ax, med, p5["x"] + 1.6, iy - 3.0, p5["w"] - 3.2, [
        "Grad-CAM no último bloco convolucional do encoder: um mapa por quadro (t−12h … t)",
        f"KAN: splines φ, importância das {cfg['dim_oculta']} variáveis de h(t), esparsidade e sobreposição entre as tarefas",
        "Spearman: variáveis latentes × Vmax, ΔVmax, IR e PMW"], fs=7.1, dy=1.36, limite_y=lim3, nome="explicabilidade")

    ax.text(1.5, 0.75, "Miniaturas do tensor (painel 1): recortes reais de 01_tensor_normal.png, gerada pelo notebook. Mapa Grad-CAM, splines e quadrados da entrada: "
            "desenhos esquemáticos, sem dados.", fontsize=5.8, color=TEXTO_2, ha="left", va="center")
    ax.set_xlim(0, W)                                                    # imshow mexe nos limites: restaura
    ax.set_ylim(0, H)
    ax.set_aspect("auto")
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--saida-pipeline", type=Path, default=RAIZ / "resultados_pipeline" / "saida_pipeline", help="pasta `saida_pipeline` extraída do zip do Kaggle")
    ap.add_argument("--analise", type=Path, default=RAIZ / "resultados_analise_pipeline", help="saída de analisar_resultados_pipeline.py")
    ap.add_argument("--saida", type=Path, default=RAIZ / "docs" / "resultado_figuras")
    ap.add_argument("--dpi", type=int, default=220)
    a = ap.parse_args()

    n = carregar_numeros(a.saida_pipeline, a.analise)
    png = a.saida_pipeline / "figuras" / "01_tensor_normal.png"
    fig = desenhar(n, png if png.exists() else None, a.dpi)
    a.saida.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.saida / "metodologia_pipeline.png", dpi=a.dpi, facecolor="white")
    fig.savefig(a.saida / "metodologia_pipeline.pdf", facecolor="white")
    print(f"figura gravada em {a.saida}/metodologia_pipeline.png (+ .pdf) | parâmetros: encoder {n['par']['encoder']}, LSTM {n['par']['lstm']}, "
          f"KAN {n['par']['kan']} × 2 = total {n['par']['total']} (confere com a execução)")
    for av in AVISOS:
        print("[aviso]", av, file=sys.stderr)
    return 1 if AVISOS else 0


if __name__ == "__main__":
    sys.exit(main())
