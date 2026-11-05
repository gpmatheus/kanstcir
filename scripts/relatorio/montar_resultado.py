#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monta `resultado--.md`: os resultados da execução completa do notebook `pipeline_tc_kaggle.ipynb` (Kaggle, 100 treinos).

Os números do texto vêm dos arquivos que o notebook gravou (`config.json`, `resumo_final.json`, `tabelas/*.csv`, `runs/*.npz`, o log do
notebook executado) e da análise independente (`analisar_resultados_pipeline.py` → `resultados_analise_pipeline/`). As poucas exceções
(os números do TCC e da rodada rápida de 2026-10-23) estão identificadas no próprio texto.

Antes de escrever, o script RECONFERE a partir das 100 predições (`runs/*.npz`): (a) as métricas de cada execução contra
`metricas_por_execucao.csv`; (b) o RMSE do ensemble das 5 seeds (Vmax e ΔVmax) contra a análise; (c) que cada janela é testada uma
única vez por (variante, seed) e que as 5 seeds de um fold testam as mesmas janelas. Se algo não bater, aborta com código 1 e não
escreve o documento.

Opcional (`--zip-baixado`, `--notebook-baixado`): confere que os arquivos que o usuário baixou do Kaggle são os mesmos que o projeto já
tinha (SHA-256 do zip; código e saída de cada célula do notebook) e registra o resultado na seção 2 do documento.

Uso (a partir da raiz do projeto; `-I` porque os arquivos baixados são dados não confiáveis: só se lê JSON, CSV e `.npz` sem pickle):
    python3 -I scripts/relatorio/figura_metodologia.py   # antes, uma vez: gera docs/resultado_figuras/metodologia_pipeline.png
    python3 -I scripts/relatorio/montar_resultado.py
    python3 -I scripts/relatorio/montar_resultado.py --zip-baixado "/caminho/resultados_pipeline (1).zip" --notebook-baixado "/caminho/kanscnn (4).ipynb"

Saída: `docs/resultado--.md` (o nome é literal, a pedido do usuário) e `docs/resultado_figuras/` (as figuras que o documento usa).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]  # raiz do projeto (este arquivo fica em scripts/relatorio/)
sys.path.insert(0, str(RAIZ / "scripts" / "relatorio"))  # o modo isolado (-I) não põe a pasta do script no caminho de importação
from figura_metodologia import carregar_numeros  # noqa: E402

DATA_DOC = "2026-10-29"
MENOS = "−"
VARIANTES = ["cnn_lstm_kan", "cnn_lstm_mlp", "cnn_kan_1frame", "cnn_lstm_kan_vmax"]
SEM_DVMAX = "cnn_lstm_kan_vmax"                      # a variante sem a cabeça de ΔVmax
ROTULO = {"cnn_lstm_kan": "`cnn_lstm_kan` (proposta)", "cnn_lstm_mlp": "`cnn_lstm_mlp`", "cnn_kan_1frame": "`cnn_kan_1frame`",
          "cnn_lstm_kan_vmax": "`cnn_lstm_kan_vmax`", "baseline_historico": "baseline de histórico (sem imagens)",
          "baseline_zero": "«prever 0» (sem imagens)", "baseline_media": "média do treino (sem imagens)"}
COMPARACOES = {"cnn_lstm_kan − cnn_lstm_mlp": "KAN × MLP (efeito da KAN)",
               "cnn_lstm_kan − cnn_kan_1frame": "LSTM × 1 quadro (efeito da sequência)",
               "cnn_lstm_kan − cnn_lstm_kan_vmax": "multitarefa × só Vmax (efeito da multitarefa)",
               "cnn_lstm_kan − baseline_historico": "rede × baseline de histórico (sem imagens)"}
VEREDITO = {"referência melhor": "proposta melhor", "referência pior": "proposta pior", "sem diferença": "sem diferença"}
TAREFA = {"vmax": "Vmax", "dvmax": "ΔVmax"}
ENS = " — ensemble das 5 seeds"
UNI = " — modelo único (média das 5 seeds)"

# figuras que o documento usa: (origem, arquivo)
FIGURAS = [("saida", "00_distribuicoes.png"), ("saida", "01_tensor_normal.png"), ("saida", "02_predito_vs_real.png"),
           ("saida", "03_rmse_e_curvas.png"), ("saida", "04_erro_por_intensidade.png"), ("saida", "05_gradcam_exemplos_vmax.png"),
           ("saida", "06_gradcam_medio_vmax.png"), ("saida", "07_kan_importancia_vmax.png"), ("saida", "08_kan_splines_vmax.png"),
           ("analise", "fig_faixa_antes_depois.png"), ("analise", "fig_201407E_quadros_longitude_trocada.png")]


# ----------------------------------------------------------------------------- formatação (vírgula decimal, "−" tipográfico)
def num(x, nd: int = 2, sinal: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    arred = round(float(x), nd)
    s = f"{abs(arred):.{nd}f}".replace(".", ",")
    if arred < 0:
        return MENOS + s
    return "+" + s if (sinal and arred > 0) else s


def inteiro(n) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def pct(x, nd: int = 1) -> str:
    return num(x, nd) + "%"


def pv(p: float) -> str:
    return "< 0,001" if p < 0.001 else num(p, 3)


def ic(lo, hi, nd: int = 2) -> str:
    return f"[{num(lo, nd, True)}; {num(hi, nd, True)}]"


def ic_abs(lo, hi, nd: int = 3) -> str:
    return f"[{num(lo, nd)}; {num(hi, nd)}]"


def dp(m, s, nd: int = 2) -> str:
    return f"{num(m, nd)} ± {num(s, nd)}"


def virgula(s) -> str:
    """'0.15 ± 0.04' (texto das tabelas do notebook) → '0,15 ± 0,04'."""
    return "—" if (s is None or (isinstance(s, float) and np.isnan(s))) else str(s).replace(".", ",")


def primeiro(s) -> str:
    """'0.15 ± 0.04' → '0,15' (só a média)."""
    return virgula(str(s).split(" ")[0])


def tabela(cab: list, linhas: list, alin: str | None = None) -> str:
    alin = alin or "l" + "r" * (len(cab) - 1)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(cab) + " |", "| " + " | ".join(sep[a] for a in alin) + " |"]
    out += ["| " + " | ".join(str(c) for c in ln) + " |" for ln in linhas]
    return "\n".join(out)


def figura(nome: str, legenda: str, largura: int | None = None) -> str:
    img = f'<img src="resultado_figuras/{nome}" alt="{legenda}" width="{largura}">' if largura else f"![{legenda}](resultado_figuras/{nome})"
    return f"{img}\n\n*{legenda}*"


def linha_da(tab: pd.DataFrame, comp: str, tar: str, mt: str) -> pd.Series:
    return tab[(tab.comparacao == comp) & (tab.tarefa == tar) & (tab.metrica == mt)].iloc[0]


# ----------------------------------------------------------------------------- leitura
def carregar(saida: Path, analise: Path) -> dict:
    t, a = saida / "tabelas", analise / "tabelas"
    d = dict(cfg=json.loads((saida / "config.json").read_text(encoding="utf-8")),
             res=json.loads((saida / "resumo_final.json").read_text(encoding="utf-8")),
             ana=json.loads((analise / "resumo_analise.json").read_text(encoding="utf-8")))
    for chave, arq in dict(met="metricas_por_execucao", boot="bootstrap_por_ciclone", fold="comparacao_por_fold", dvb="dvmax_modelos_vs_baselines",
                           base="baselines_dvmax_por_fold", wexec="wilcoxon_por_execucao", wjan="wilcoxon_por_janela").items():
        d[chave] = pd.read_csv(t / f"{arq}.csv")
    for chave, arq in dict(resumo="resumo_por_variante", ens="dvmax_modelo_unico_ensemble_empilhamento", boot_ens="bootstrap_ensemble",
                           ri="ri_auc_ap_bootstrap", inten="erro_por_intensidade", r2017="resultados_2017", sobre="sobreajuste_por_variante",
                           varia="variacao_entre_folds_e_seeds", sens="sensibilidade_metricas", boot_sem="bootstrap_sem_afetadas",
                           faixa_lim="faixa_por_limite", faixa_cic="faixa_composicao_por_ciclone").items():
        d[chave] = pd.read_csv(a / f"{arq}.csv")
    return d


# ----------------------------------------------------------------------------- verificação independente (a partir das 100 predições)
def metricas_reg(y: np.ndarray, p: np.ndarray) -> tuple:
    e = p - y
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e))), float(np.mean(e))


def verificar_npz(saida: Path, met: pd.DataFrame, resumo: pd.DataFrame, ens: pd.DataFrame, n_janelas: int, tol: float = 1e-4) -> dict:
    runs = sorted((saida / "runs").glob("*.npz"))
    if len(runs) != len(met):
        raise SystemExit(f"[erro] {len(runs)} arquivos .npz em runs/, mas a tabela tem {len(met)} execuções")
    tab = met.set_index(["variante", "fold", "seed"])
    dif = {c: 0.0 for c in ["rmse_vmax", "mae_vmax", "vies_vmax", "rmse_dvmax", "mae_dvmax"]}
    grupos: dict = {}
    for p in runs:
        v, f, s, _ = p.stem.split("__")
        fold, seed = int(f[1:]), int(s[1:])
        with np.load(p, allow_pickle=False) as z:                  # nunca pickle: os arquivos baixados são dados não confiáveis
            idx = z["idx_teste"].astype(np.int64)
            yv, pvm = z["y_vmax"].astype(np.float64), z["pred_vmax"].astype(np.float64)
            yd, pdm = z["y_dvmax"].astype(np.float64), z["pred_dvmax"].astype(np.float64)
        linha = tab.loc[(v, fold, seed)]
        r, m, vi = metricas_reg(yv, pvm)
        calc = dict(rmse_vmax=r, mae_vmax=m, vies_vmax=vi)
        ok = np.isfinite(yd) & np.isfinite(pdm)
        if ok.any() and np.isfinite(linha["rmse_dvmax"]):
            r2, m2, _ = metricas_reg(yd[ok], pdm[ok])
            calc.update(rmse_dvmax=r2, mae_dvmax=m2)
        for c, x in calc.items():
            dif[c] = max(dif[c], abs(x - float(linha[c])))
        grupos.setdefault((v, fold), []).append((seed, idx, yv, yd, pvm, pdm))

    # cada janela é testada exatamente uma vez por (variante, seed)
    particao_ok = True
    for v in VARIANTES:
        for seed in sorted({t[0] for fold in range(5) for t in grupos[(v, fold)]}):
            todas = np.concatenate([[t for t in grupos[(v, fold)] if t[0] == seed][0][1] for fold in range(5)])
            particao_ok &= (len(todas) == n_janelas) and (len(np.unique(todas)) == n_janelas)

    # ensemble das 5 seeds: média das previsões por janela (as 5 seeds de um fold testam as MESMAS janelas, na mesma ordem).
    # Convenções da análise (`analisar_resultados_pipeline.py`): o RMSE de Vmax do ensemble é calculado com as 65.670 janelas juntas
    # ("agrupado"); o RMSE/MAE de ΔVmax é a média dos valores dos 5 folds (a convenção das tabelas do notebook). Calcula-se também a outra
    # convenção de cada um, só para dizer no documento quanto elas diferem.
    rmse_ens, dif_ens = {}, 0.0
    for v in VARIANTES:
        Y, P, YD, PD, r_f, r_fd, m_fd = [], [], [], [], [], [], []
        for fold in range(5):
            g = sorted(grupos[(v, fold)], key=lambda t: t[0])
            for _, idx, y, yd, _, _ in g[1:]:
                if not (np.array_equal(idx, g[0][1]) and np.array_equal(y, g[0][2]) and np.array_equal(yd, g[0][3], equal_nan=True)):
                    raise SystemExit(f"[erro] as seeds de ({v}, fold {fold}) não testam as mesmas janelas")
            p_f, pd_f = np.mean([t[4] for t in g], axis=0), np.mean([t[5] for t in g], axis=0)
            Y.append(g[0][2]); YD.append(g[0][3]); P.append(p_f); PD.append(pd_f)
            r_f.append(metricas_reg(g[0][2], p_f)[0])
            if v != SEM_DVMAX:
                ok = np.isfinite(g[0][3])
                rr, mm, _ = metricas_reg(g[0][3][ok], pd_f[ok])
                r_fd.append(rr); m_fd.append(mm)
        Y, P, YD, PD = map(np.concatenate, (Y, P, YD, PD))
        rmse_ens[v] = dict(vmax=metricas_reg(Y, P)[0], vmax_media_folds=float(np.mean(r_f)))
        dif_ens = max(dif_ens, abs(rmse_ens[v]["vmax"] - float(resumo.set_index("variante").loc[v, "rmse_vmax_ensemble_seeds"])))
        if v != SEM_DVMAX:
            ok = np.isfinite(YD)
            rmse_ens[v].update(dvmax=float(np.mean(r_fd)), mae_dvmax=float(np.mean(m_fd)), dvmax_agrupado=metricas_reg(YD[ok], PD[ok])[0])
            ref = ens.set_index("modelo").loc[v + ENS]
            dif_ens = max(dif_ens, abs(rmse_ens[v]["dvmax"] - float(ref["rmse"])), abs(rmse_ens[v]["mae_dvmax"] - float(ref["mae"])))
    pior = max(max(dif.values()), dif_ens)
    return dict(execucoes=len(runs), dif_por_metrica=dif, dif_ensemble=dif_ens, particao_ok=bool(particao_ok), n_janelas=n_janelas, tolerancia=tol,
                ok=bool(particao_ok and pior <= tol), rmse_ensemble=rmse_ens)


# ----------------------------------------------------------------------------- notebook executado e arquivos baixados
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def texto_saida(c: dict) -> str:
    partes = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            partes.append("".join(o.get("text", [])))
        elif "data" in o:
            partes.append("".join(o["data"].get("text/plain", [])))
        elif o.get("output_type") == "error":
            partes.append("ERRO " + o.get("ename", ""))
    return "".join(partes)


def papermill_de(nb: dict) -> dict:
    """Início, fim e duração (s) que o papermill do Kaggle gravou nos metadados do notebook; None onde faltar."""
    pm = nb.get("metadata", {}).get("papermill", {})
    return dict(inicio=pm.get("start_time"), fim=pm.get("end_time"), duracao=pm.get("duration"))


def quando(iso: str | None) -> str:
    """'2026-10-12T20:38:14.958846+00:00' → '2026-10-12 20:38:14 UTC' (o Kaggle grava em UTC)."""
    if not iso:
        return "—"
    t = datetime.fromisoformat(iso).astimezone(timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def horas_min(segundos: float) -> str:
    h, m = divmod(int(round(segundos / 60)), 60)
    return f"{h} h {m:02d} min"


def ler_notebook(nb_zip: Path, raiz: Path, zip_baixado: Path | None, nb_baixado: Path | None) -> dict:
    """Lê o notebook executado como JSON (nunca o executa) e, se pedido, confere os arquivos baixados contra o que o projeto já tinha."""
    nb = json.loads(nb_zip.read_text(encoding="utf-8"))
    cel = nb["cells"]
    info: dict = dict(papermill=papermill_de(nb), celulas=(len(cel), sum(c["cell_type"] == "code" for c in cel)),
                      erros=sum(o.get("output_type") == "error" for c in cel for o in c.get("outputs", [])),
                      stderr=sum(o.get("name") == "stderr" for c in cel for o in c.get("outputs", [])))
    # a tabela de 2017 do próprio notebook (usa só a 1ª seed): linhas "i  variante  RMSE MAE N RMSE17 MAE17 N17"
    info["nb2017"] = {}
    for c in cel:
        t = texto_saida(c) if c["cell_type"] == "code" else ""
        if "RMSE (só 2017)" in t:
            for m in re.finditer(r"^\s*\d+\s+(cnn_\w+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", t, re.M):
                info["nb2017"][m.group(1)] = dict(rmse=float(m.group(5)), mae=float(m.group(6)), n=int(m.group(7)))
    zip_proj = raiz / "resultados_pipeline.zip"
    if zip_baixado is not None and zip_baixado.exists() and zip_proj.exists():
        a, b = sha256(zip_baixado), sha256(zip_proj)
        info["zip"] = dict(igual=(a == b), sha=a, bytes=zip_baixado.stat().st_size)
    if nb_baixado is not None and nb_baixado.exists():
        nb_b = json.loads(nb_baixado.read_text(encoding="utf-8"))
        a = nb_b["cells"]
        info["notebook"] = dict(celulas=len(a), mesmas_celulas=(len(a) == len(cel)),
                                dif_fonte=[i for i, (x, y) in enumerate(zip(a, cel)) if "".join(x["source"]) != "".join(y["source"])],
                                dif_saida=[i for i, (x, y) in enumerate(zip(a, cel)) if x["cell_type"] == "code" and texto_saida(x) != texto_saida(y)],
                                sem_saida_no_zip=[i for i, (x, y) in enumerate(zip(a, cel)) if x["cell_type"] == "code" and texto_saida(x) and not texto_saida(y)])
        pm_b = papermill_de(nb_b)
        # a cópia do notebook que vai dentro do zip é gravada ENQUANTO a última célula empacota a saída, então ali o fim e a duração
        # ainda não existem; o arquivo baixado depois traz o registro completo (mesmo início)
        if pm_b["inicio"] == info["papermill"]["inicio"] and not info["papermill"]["fim"] and pm_b["fim"]:
            info["papermill"] = pm_b
    return info


# ----------------------------------------------------------------------------- blocos do documento
def camadas_encoder(cfg: dict) -> list:
    """(descrição, saída, parâmetros) de cada camada do encoder; mesmas contas de `figura_metodologia.contar_parametros`."""
    linhas, c_in, lado = [], 4, cfg["recorte_final"]
    for c in cfg["canais_cnn"]:
        lado //= 2
        linhas.append((f"Conv 3×3, passo 2, {c} canais + BatchNorm + ReLU", f"({c}, {lado}, {lado})", c_in * c * 9 + 2 * c))
        c_in = c
    d = cfg["dim_embedding"]
    linhas.append((f"Flatten → Dropout({num(cfg['dropout'], 1)}) → Linear({c_in * lado * lado} → {d}) → LayerNorm", f"({d})", c_in * lado * lado * d + d + 2 * d))
    return linhas


def valores_comuns(d: dict, n: dict, ens_val: dict) -> dict:
    """Números que aparecem em mais de uma seção (todos calculados dos arquivos)."""
    met, ens = d["met"], d["ens"].set_index("modelo")
    ref = met[met.variante == "cnn_lstm_kan"]
    hist = d["base"][d["base"].variante == "baseline_historico"]
    nf = d["fold"]
    return dict(
        ref_rmse=ref["rmse_vmax"].mean(), ref_dp=ref["rmse_vmax"].std(), ref_n=len(ref), ref_dvmax=ref["rmse_dvmax"].mean(),
        m1={v: met[met.variante == v]["rmse_vmax"].mean() for v in VARIANTES},
        hist_rmse=hist["rmse_dvmax"].mean(), prev=100 * n["ri"] / n["com_dvmax"],
        e_kan=ens.loc["cnn_lstm_kan" + ENS], u_kan=ens.loc["cnn_lstm_kan" + UNI],
        wexec=lambda v, mt: d["wexec"][(d["wexec"].comparacao == f"cnn_lstm_kan × {v}") & (d["wexec"].metrica == mt)].iloc[0],
        fold=lambda comp, tar, mt: linha_da(nf, comp, tar, mt), ens_val=ens_val)


def secao_resumo(d: dict, n: dict, b: pd.DataFrame, bens: pd.DataFrame, c: dict) -> str:
    met, ri, ens = d["met"], d["ri"].set_index("escore"), d["ens"].set_index("modelo")
    r = lambda comp, tar, mt, tab=b: linha_da(tab, comp, tar, mt)                       # noqa: E731
    lstm, kan_v, kan_d, mt_, hist = (r("cnn_lstm_kan − cnn_kan_1frame", "vmax", "rmse"), r("cnn_lstm_kan − cnn_lstm_mlp", "vmax", "rmse"),
                                     r("cnn_lstm_kan − cnn_lstm_mlp", "dvmax", "rmse"), r("cnn_lstm_kan − cnn_lstm_kan_vmax", "vmax", "rmse"),
                                     r("cnn_lstm_kan − baseline_historico", "dvmax", "rmse"))
    kan_de, hist_e = r("cnn_lstm_kan − cnn_lstm_mlp", "dvmax", "rmse", bens), r("cnn_lstm_kan − baseline_historico", "dvmax", "rmse", bens)
    auc_e, auc_h = ri.loc["cnn_lstm_kan (ensemble)"], ri.loc["baseline_historico"]
    pilha, pilha_e = ens.loc["histórico + cnn_lstm_kan — modelo único (média das seeds)"], ens.loc["histórico + cnn_lstm_kan — ensemble"]
    f17, fx = d["r2017"].set_index("modelo"), d["ana"]["faixa"]
    dvb = d["dvb"].set_index("modelo").loc["cnn_lstm_kan"]
    fo = c["fold"]
    mt_folds = 5 - int(fo("cnn_lstm_kan − cnn_lstm_kan_vmax", "vmax", "rmse").folds_ref_melhor.split("/")[0])
    linhas = [
        ("A sequência de quadros (LSTM) ajuda?", "**Sim, é o efeito mais forte.**",
         f"RMSE de Vmax {num(lstm.dif, sinal=True)} nó {ic(lstm.ic95_inf, lstm.ic95_sup)} contra a variante de 1 quadro ({num(100 * lstm.dif / c['m1']['cnn_kan_1frame'], 0, True)}%); "
         f"a proposta é melhor em {fo('cnn_lstm_kan − cnn_kan_1frame', 'vmax', 'rmse').folds_ref_melhor} folds"),
        ("A KAN é melhor que um MLP?", "**Não dá para afirmar.**",
         f"Vmax: empate ({num(kan_v.dif, 3, True)} {ic(kan_v.ic95_inf, kan_v.ic95_sup, 3)}; p Holm {pv(kan_v.p_holm)}). ΔVmax: {num(kan_d.dif, sinal=True)} nó "
         f"(p Holm {pv(kan_d.p_holm)}), mas em {fo('cnn_lstm_kan − cnn_lstm_mlp', 'dvmax', 'rmse').folds_ref_melhor} folds e, com o ensemble das seeds, "
         f"{num(kan_de.dif, sinal=True)} (p Holm {pv(kan_de.p_holm)}: não significativo)"),
        ("A multitarefa (Vmax + ΔVmax) ajuda o Vmax?", "**Não: piora levemente.**",
         f"RMSE de Vmax {num(mt_.dif, sinal=True)} nó {ic(mt_.ic95_inf, mt_.ic95_sup)}; a variante só-Vmax é melhor em {mt_folds}/5 folds"),
        ("Em ΔVmax a rede supera um baseline sem imagens?", "**Só o ensemble de 5 seeds.**",
         f"modelo único: {num(hist.dif, sinal=True)} nó {ic(hist.ic95_inf, hist.ic95_sup)} (empate); ensemble: RMSE {num(c['e_kan'].rmse)} contra {num(c['hist_rmse'])} "
         f"({num(hist_e.dif, sinal=True)} {ic(hist_e.ic95_inf, hist_e.ic95_sup)})"),
        ("E na intensificação rápida (RI)?", "**A rede ordena melhor os casos; o alarme com limiar de 30 nós quase não funciona.**",
         f"ensemble × histórico: AUC {num(auc_e.auc, 3)} × {num(auc_h.auc, 3)} e AP {num(auc_e.ap, 3)} × {num(auc_h.ap, 3)} "
         f"(AUC {num(auc_h.dif_auc_vs_ref, 3, True)} {ic(auc_h.dif_auc_ic_inf, auc_h.dif_auc_ic_sup, 3)}); com o limiar fixo de 30 nós, POD {primeiro(dvb.pod_ri)} e FAR {primeiro(dvb.far_ri)}"),
        ("Imagens e histórico de Vmax se complementam?", "**Indicativo (análise exploratória).**",
         f"empilhando os dois, RMSE de ΔVmax {num(pilha.rmse)} (ensemble: {num(pilha_e.rmse)}) contra {num(c['hist_rmse'])} só com o histórico"),
        ("Comparação com o TCC (teste de 2017)?", "**Melhor, com ressalvas.**",
         f"RMSE {num(f17.loc['cnn_lstm_kan', 'rmse_2017'])} ± {num(f17.loc['cnn_lstm_kan', 'rmse_dp_seeds'])} / MAE {num(f17.loc['cnn_lstm_kan', 'mae_2017'])} contra "
         f"{num(f17.loc['CNN-TC (TCC)', 'rmse_2017'])} / {num(f17.loc['CNN-TC (TCC)', 'mae_2017'])} (CNN-TC) e {num(f17.loc['MobileNetV2+SDTA (TCC)', 'rmse_2017'])} / "
         f"{num(f17.loc['MobileNetV2+SDTA (TCC)', 'mae_2017'])} (MobileNetV2+SDTA); o protocolo é outro (seção 4.7)"),
        ("Há problema nos dados?", "**Sim: um ciclone com a longitude gravada errada.**",
         f"`201407E`: {fx['janelas_afetadas']} janelas ({pct(100 * fx['janelas_afetadas'] / n['janelas'], 2)} do total) respondem por {pct(fx['pct_sq_afetadas'])} do erro "
         f"quadrático e produzem a faixa horizontal do gráfico predito × real (seção 4.8)"),
    ]
    t = tabela(["Pergunta", "Resposta", "Evidência (erros em nós; diferença = proposta − outra, negativa = a proposta erra menos)"], linhas, "lll")
    return f"""## 1. Resumo

Rodou-se no Kaggle ({n['gpu']}) o notebook `pipeline_tc_kaggle.ipynb`: **{n['n_exec']} treinos** ({n['n_folds']} folds × {n['n_seeds']} seeds × {n['n_var']} variantes), \
{num(n['horas'], 2)} h, com {d['res']['execucoes_pendentes']} execuções pendentes. O modelo proposto (CNN encoder com pesos compartilhados → LSTM → duas cabeças KAN) estima o **Vmax com RMSE de \
{num(c['ref_rmse'])} ± {num(c['ref_dp'])} nós** (modelo único; média das {c['ref_n']} execuções fold × seed) e **{num(c['ens_val']['cnn_lstm_kan']['vmax'])} nós com o ensemble das 5 seeds**.

{t}

**Três cuidados ao citar estes números**

1. **Modelo único × ensemble.** O ensemble das 5 seeds é outro modelo (5× o custo de treino e de inferência) e ganha {num(c['ref_rmse'] - c['ens_val']['cnn_lstm_kan']['vmax'])} nó em Vmax e \
{num(c['ref_dvmax'] - c['e_kan'].rmse)} nó em ΔVmax; as conclusões de ΔVmax e de RI mudam entre os dois. Diga sempre qual dos dois está sendo reportado.
2. **A comparação com o TCC não é estrita.** Aqui a validação é cruzada por ciclone (todos os anos) e as janelas exigem 7 quadros consecutivos; o TCC treinou em 2003–2014 e testou em 2017 (4.296 imagens).
3. **O erro de Vmax carrega um defeito dos dados.** Sem as {fx['janelas_afetadas']} janelas do `201407E`, o RMSE do ensemble cai de {num(fx['rmse_todas'])} para {num(fx['rmse_sem_afetadas'])}; nenhuma conclusão muda.
"""


def secao_execucao(d: dict, n: dict, nb: dict) -> str:
    res, lg = d["res"], d["ana"]["log"]
    dup, pm = lg["duplicatas"], nb["papermill"]
    if pm["inicio"] and pm["fim"]:
        relogio = f"{quando(pm['inicio'])} → {quando(pm['fim'])}" + (f" ({horas_min(pm['duracao'])} no total, das quais {num(n['horas'], 2)} h de treino)" if pm["duracao"] else "")
    else:
        relogio = f"início em {quando(pm['inicio'])} (o fim não está registrado no notebook do zip)" if pm["inicio"] else "—"
    linhas = [("Notebook", "`pipeline_tc_kaggle.ipynb` (o notebook **completo**, em oposição ao notebook rápido de 1 seed), na versão ajustada depois da 1ª execução no Kaggle; "
                           f"{nb['celulas'][0]} células, {nb['celulas'][1]} de código"),
              ("Ambiente", f"Kaggle, {n['gpu']} ({num(lg['gpu']['memoria_gb'], 1)} GB), Python {lg['ambiente']['python']}, PyTorch {lg['ambiente']['torch']}"),
              ("Treino", f"{n['n_exec']} execuções de {lg['tempo_treino']['inicio']} a {lg['tempo_treino']['fim']} (**{num(n['horas'], 2)} h**), {res['execucoes_pendentes']} pendentes; "
                         f"assinatura da configuração `{n['assinatura']}`"),
              ("Execução no Kaggle (papermill)", relogio),
              ("Erros e avisos no log", f"{nb['erros']} células com erro; {nb['stderr']} saídas em stderr"),
              ("Dados", f"{inteiro(n['imagens'])} imagens, {inteiro(n['ids'])} IDs → {inteiro(res['dados']['ciclones'])} ciclones depois de unir {dup['ids_unificados']} IDs; "
                        f"{dup['registros_removidos']} duplicatas removidas ({dup['identicos_bit_a_bit']} idênticas bit a bit); {n['invalidos']} quadros inválidos"),
              ("Janelas", f"**{inteiro(n['janelas'])}** de {inteiro(n['ciclones'])} ciclones; {inteiro(n['com_dvmax'])} com rótulo de ΔVmax; "
                          f"RI (ΔVmax ≥ 30 nós em 24 h) em {inteiro(n['ri'])} ({pct(100 * n['ri'] / n['com_dvmax'])} das rotuladas)")]
    extra = []
    if "zip" in nb:
        z = nb["zip"]
        extra.append(f"- **`resultados_pipeline (1).zip`** ({inteiro(z['bytes'])} bytes): {'**idêntico byte a byte ao**' if z['igual'] else '**diferente do**'} `resultados_pipeline.zip` que o projeto já tinha "
                     f"(SHA-256 `{z['sha'][:12]}…{z['sha'][-8:]}`). " + ("Não há dados novos em relação à análise de 2026-10-16." if z["igual"] else "Conferir antes de usar."))
    if "notebook" in nb:
        k = nb["notebook"]
        ds = k["dif_saida"]
        plural = len(ds) > 1
        if not ds:
            saida = "o texto de saída é o mesmo em todas as células"
        else:
            saida = f"o texto de saída é o mesmo, exceto na{'s' if plural else ''} célula{'s' if plural else ''} {', '.join(map(str, ds))}"
            if ds == k["sem_saida_no_zip"] and ds[-1] == k["celulas"] - 1:
                saida += (" (a última, que empacota a saída: na cópia dentro do zip ela ainda não tem saída, porque o próprio zip é criado por ela; "
                          "o mesmo vale para o fim do registro do papermill)")
        fonte = "o código e o texto de todas as células são iguais aos do" if not k["dif_fonte"] else f"o código ou o texto das células {k['dif_fonte']} diferem do"
        extra.append(f"- **`kanscnn (4).ipynb`**: {k['celulas']} células; {fonte} `__notebook__.ipynb` do zip; {saida}. "
                     "É o **mesmo notebook executado**: a 2ª execução no Kaggle (a 1ª, de 75 treinos, foi feita antes e não faz parte deste documento).")
    return "## 2. A execução\n\n" + tabela(["Item", "Valor"], linhas, "ll") + ("\n\n" + "\n".join(extra) if extra else "") + "\n"


def secao_metodologia(d: dict, n: dict, c: dict) -> str:
    cfg, par = d["cfg"], n["par"]
    enc = camadas_encoder(cfg)
    assert sum(x[2] for x in enc) == par["encoder"], "contas por camada ≠ total do encoder"
    camadas = [(a, b, inteiro(p)) for a, b, p in enc]
    camadas += [(f"LSTM ({cfg['dim_oculta']} unidades) sobre os {cfg['tam_janela']} vetores e(t−12 h … t); estado final h(t) ∈ (−1, 1)", f"({cfg['dim_oculta']})", inteiro(par["lstm"])),
                (f"2 × cabeça KAN [{cfg['dim_oculta']} → {' → '.join(map(str, cfg['kan_ocultas']))} → 1] (B-splines de grau {cfg['kan_ordem']}, grid {cfg['kan_grid']}, "
                 f"{par['funcoes']} funções-base por aresta)", "(1) cada", f"{inteiro(par['kan'])} cada")]
    var = [("`cnn_lstm_kan` (proposta)", "LSTM", "KAN", "Vmax + ΔVmax", "—"), ("`cnn_lstm_mlp`", "LSTM", f"MLP [{cfg['dim_oculta']}→{cfg['mlp_ocultas'][0]}→1]", "Vmax + ΔVmax", "a KAN"),
           ("`cnn_kan_1frame`", "nenhuma (só o quadro t)", "KAN", "Vmax + ΔVmax", "a sequência / a LSTM"), ("`cnn_lstm_kan_vmax`", "LSTM", "KAN", "só Vmax", "a multitarefa")]
    var = [(*v, inteiro(n["parametros"][k])) for v, k in zip(var, VARIANTES)]
    hp = [("Otimizador", f"AdamW (lr de pico {num(cfg['lr'] * 1000, 0)}×10⁻³, weight decay {num(cfg['weight_decay'] * 1e4, 0)}×10⁻⁴)"),
          ("Agenda", "OneCycle (aquece nos primeiros 15% dos passos e decai até quase zero)"),
          ("Épocas, lote", f"até {cfg['epocas_max']} épocas, lote {cfg['tam_lote']}, corte de gradiente em norma {num(cfg['clip_grad'], 1)}"),
          ("Precisão", "mista (float16 com GradScaler) no encoder e na LSTM; **cabeças KAN sempre em float32**" if cfg["usar_amp"] else "float32"),
          ("Perda", f"{num(cfg['pesos_tarefas']['vmax'], 1)}·MSE(Vmax) + {num(cfg['pesos_tarefas']['dvmax'], 1)}·MSE(ΔVmax) sobre alvos padronizados; o termo de ΔVmax só nas janelas com rótulo em t+{cfg['horizonte_horas']} h"),
          ("Early stopping", f"RMSE de Vmax na validação interna ({int(cfg['frac_validacao_interna'] * 100)}% dos ciclones do treino, inteiros); paciência {cfg['paciencia']}; guarda os melhores pesos"),
          ("Validação", f"cruzada agrupada por ciclone, k = {cfg['n_folds']} (nenhum ciclone aparece em dois folds); {n['n_seeds']} seeds por fold; os mesmos folds para todas as variantes (comparação pareada)"),
          ("Aumento de dados", f"rotação aleatória de 0° a 360° com probabilidade {num(cfg['prob_rotacao'], 2)}, só no treino, **a mesma rotação nos 5 quadros e nos 4 canais**; DTA e SDTA calculados antes de girar")]
    return f"""## 3. Metodologia

{figura('metodologia_pipeline.png', 'Metodologia do experimento (gerada por figura_metodologia.py com os números da própria execução). As miniaturas do tensor são recortes reais; Grad-CAM e splines são esquemas.')}

### 3.1 Dados e janelas

- **TCIR** ({inteiro(n['imagens'])} imagens de 201×201 px, 2003–2017): usam-se as bandas **IR** (temperatura do topo das nuvens) e **PMW** (chuva por micro-ondas, que enxerga a parede do olho através das nuvens); WV e VIS são descartadas (WV ≈ IR; VIS é `NaN` à noite).
- **Limpeza.** `NaN` e valores > {inteiro(cfg['invalido_acima_de'])} viram 0 (como no TCC); {n['dup_removidas']} registros duplicados foram removidos e {n['ids_unificados']} IDs unificados (o mesmo ciclone com um ID em cada bacia), \
para que as cópias nunca caiam em folds diferentes; quadros com mais de {int(cfg['max_frac_nan_ir'] * 100)}% de IR ausente ({n['invalidos']}) saem da série.
- **Janelas.** Cada janela é uma sequência de {cfg['tam_janela']} quadros a cada {cfg['passo_horas']} h (t−12 h … t); como o SDTA do quadro mais antigo precisa de dois quadros anteriores, leem-se {cfg['tam_janela'] + 2} consecutivos. \
Nenhuma janela atravessa um buraco. Resultado: **{inteiro(n['janelas'])} janelas** de {inteiro(n['ciclones'])} ciclones.
- **Alvos.** Vmax(t) em nós e ΔVmax = Vmax(t+{cfg['horizonte_horas']} h) − Vmax(t); o valor futuro é só rótulo, **nunca entrada**. RI = ΔVmax ≥ {int(cfg['limiar_ri_nos'])} nós.
- **Tensor por quadro** `(4, {cfg['recorte_final']}, {cfg['recorte_final']})`: IR, PMW, **DTA** (diferença absoluta entre quadros consecutivos: a «velocidade» de mudança da nuvem) e **SDTA** (a diferença das diferenças: a «aceleração»); \
z-score por canal com μ e σ só do treino de cada fold; recorte central {cfg['recorte_armazenado']}→{cfg['recorte_final']} px.

{figura('00_distribuicoes.png', 'Distribuição do Vmax(t), do ΔVmax e do número de janelas por ano (figura 00 do notebook).')}

{figura('01_tensor_normal.png', 'Exemplo de janela (ciclone 201007W, sem rotação): 5 quadros × 4 canais (figura 01 do notebook).', 760)}

### 3.2 Modelo

O **mesmo** encoder processa os {cfg['tam_janela']} quadros (pesos compartilhados: uma só forma de descrever um quadro de ciclone); a **LSTM** liga as 5 descrições numa história e entrega o estado final h(t); cada **cabeça KAN** lê h(t) e prevê uma grandeza. \
Numa KAN, a não linearidade mora nas **arestas** (cada uma carrega uma função φ aprendível, aqui uma B-spline), não nos nós; por isso cada φ pode ser desenhada.

{tabela(['Bloco', 'Saída', 'Parâmetros'], camadas, 'lrr')}

Total da proposta: **{inteiro(par['total'])} parâmetros** (conferido contra o gravado pela execução).

### 3.3 Variantes (ablações)

{tabela(['Variante', 'Modelagem temporal', 'Cabeça', 'Tarefas', 'Isola o efeito de', 'Parâmetros'], var, 'lllllr')}

### 3.4 Treino

{tabela(['Item', 'Valor'], hp, 'll')}

### 3.5 Avaliação e estatística

- **Métricas** (em nós): RMSE (penaliza erros grandes), MAE e viés de Vmax e de ΔVmax. **RI**: POD, FAR e CSI com o limiar fixo de {int(cfg['limiar_ri_nos'])} nós e com um limiar **ajustado na validação interna** (o teste nunca é consultado), além de **AUC** e **AP**, que usam o próprio ΔVmax previsto como escore e não dependem de limiar.
- **Baselines sem imagens para ΔVmax**: «prever 0», a média do treino e o **histórico de Vmax** (regressão polinomial de grau 2 no Vmax real de t e nas tendências de 3, 6 e 12 h). Atenção: o baseline de histórico usa uma informação que a rede **não recebe**.
- **Teste principal: bootstrap por ciclone** ({inteiro(cfg['bootstrap_reps'])} réplicas), com correção de Holm-Bonferroni sobre os 12 testes. A unidade independente é o **ciclone** ({inteiro(n['ciclones'])}), não a janela (janelas vizinhas do mesmo ciclone são quase iguais) nem a seed \
(as 5 seeds de um fold testam as mesmas janelas). Convenção: diferença = proposta − outra; **negativa = a proposta erra menos**. É convincente quando o intervalo de 95% não contém 0 e a direção se repete nos folds.
- **Apoio (não decide):** 1 ponto por fold (com 5 pontos o menor p do Wilcoxon exato é {num(d['fold']['menor_p_wilcoxon'].iloc[0], 4)}), Wilcoxon por janela ({len(d['wjan'])} testes, p-valores otimistas) e Wilcoxon por execução (exploratório: os {n['n_folds'] * n['n_seeds']} pares valem ≈ {n['n_folds']} observações, o que infla a significância).
- **Ensemble das 5 seeds**: média das previsões das 5 seeds de cada fold; é um modelo diferente e mais caro.
- **Explicabilidade**: **Grad-CAM** no último bloco convolucional do encoder (um mapa por quadro; {cfg['gradcam_amostras']} janelas de teste no mapa médio) e, na KAN, as splines, a importância das {cfg['dim_oculta']} variáveis de h(t), a esparsidade e a correlação de Spearman das variáveis latentes com grandezas físicas.
"""


def linha_modelo(met: pd.DataFrame, v: str) -> list:
    g = met[met.variante == v]
    dv = bool(g["rmse_dvmax"].notna().all())
    s = lambda col: dp(g[col].mean(), g[col].std())                                     # noqa: E731
    return [ROTULO[v], s("rmse_vmax"), s("mae_vmax"), s("rmse_dvmax") if dv else "—", s("mae_dvmax") if dv else "—",
            num(g["auc_ri"].mean()) if dv else "—", num(g["ap_ri"].mean()) if dv else "—", num(g["csi_ri_aj"].mean()) if dv else "—",
            num(g["melhor_epoca"].mean(), 1), f"{num(g['segundos'].mean() / 60, 1)} min"]


def secao_resultados(d: dict, n: dict, c: dict, nb: dict) -> str:
    met, base, boot, fold = d["met"], d["base"], d["boot"], d["fold"]
    ens, rv, ens_val, cfg = d["ens"].set_index("modelo"), d["resumo"].set_index("variante"), c["ens_val"], d["cfg"]
    out = ["## 4. Resultados", "",
           "Como ler: erros em **nós**; menor é melhor. «Modelo único» = média ± desvio entre as 25 execuções (fold × seed); em cada execução o teste é um fold inteiro de ciclones que o modelo não viu.", ""]

    # 4.1 desempenho
    linhas = [linha_modelo(met, v) for v in VARIANTES]
    for k in ("baseline_historico", "baseline_zero", "baseline_media"):
        g = base[base.variante == k]
        linhas.append([ROTULO[k], "—", "—", dp(g.rmse_dvmax.mean(), g.rmse_dvmax.std()), dp(g.mae_dvmax.mean(), g.mae_dvmax.std()),
                       num(g.auc_ri.mean()), num(g.ap_ri.mean()), num(g.csi_ri_aj.mean()), "—", "—"])
    out += ["### 4.1 Desempenho por variante (modelo único)", "",
            tabela(["Variante", "Vmax RMSE", "Vmax MAE", "ΔVmax RMSE", "ΔVmax MAE", "RI AUC", "RI AP", "RI CSI (limiar ajustado)", "Melhor época (média)", "Tempo por treino"], linhas, "lrrrrrrrrr"),
            "", "Os baselines sem imagens não têm seeds: a média ± desvio é entre os 5 folds. O AP do baseline de histórico pode sofrer o defeito dos empates na ordenação (a corrigir; o efeito não foi medido). "
            "A variante só-Vmax não tem a cabeça de ΔVmax.", "",
            figura("02_predito_vs_real.png", "Vmax previsto × real, todas as janelas, seed 0 (os títulos mostram o RMSE da seed 0 e por isso diferem alguns centésimos das médias de 25 execuções da tabela). "
                   "A faixa horizontal em ≈ 25 nós é explicada na seção 4.8."), ""]

    # 4.2 por fold e entre seeds
    v = d["varia"].set_index("variante")
    linhas = [[ROTULO[k]] + [num(v.loc[k, f"fold_{i}"]) for i in range(5)] + [num(v.loc[k, "entre_folds"]), num(v.loc[k, "entre_seeds"])] for k in VARIANTES]
    f_ref = [v.loc["cnn_lstm_kan", f"fold_{i}"] for i in range(5)]
    pior = int(np.argmax(f_ref))
    n_pior = sum(int(np.argmax([v.loc[k, f'fold_{i}'] for i in range(5)])) == pior for k in VARIANTES)
    out += ["### 4.2 Por fold: o fold pesa mais que a seed", "",
            tabela(["RMSE de Vmax (média das 5 seeds)", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "dp entre folds", "dp entre seeds"], linhas, "lrrrrrrr"), "",
            f"O **fold {pior} é o mais difícil** (na proposta, {num(f_ref[pior] - np.mean([x for i, x in enumerate(f_ref) if i != pior]), 1, True)} nó acima da média dos outros) e é o pior em "
            f"{'todas as ' + str(len(VARIANTES)) if n_pior == len(VARIANTES) else str(n_pior) + ' das ' + str(len(VARIANTES))} variantes. Cada fold tem {inteiro(n['janelas'] / n['n_folds'])} janelas e ≈ {inteiro(n['ciclones'] / n['n_folds'])} ciclones. "
            "A diferença entre variantes é muito maior que o ruído de seed, mas o fold desloca todas de uma vez.", "",
            figura("03_rmse_e_curvas.png", f"RMSE de Vmax (esquerda) e de ΔVmax (centro) por execução: uma bolinha = um fold × seed; no painel de Vmax, o grupo de bolinhas mais alto de cada variante é o fold {pior}. "
                   "À direita, o RMSE de Vmax na validação interna a cada época, média ± desvio das execuções (figura 03 do notebook)."), ""]

    # 4.3 ensemble
    linhas = []
    for k in VARIANTES:
        e, tem = ens_val[k], k != SEM_DVMAX
        u = ens.loc[k + UNI] if tem else None
        linhas.append([ROTULO[k], num(rv.loc[k, "rmse_vmax"]), num(e["vmax"]), num(rv.loc[k, "rmse_vmax"] - e["vmax"]),
                       num(u.rmse) if tem else "—", num(e["dvmax"]) if tem else "—", num(u.rmse - e["dvmax"]) if tem else "—",
                       num(ens.loc[k + ENS, "auc"], 3) if tem else "—", num(ens.loc[k + ENS, "ap"], 3) if tem else "—"])
    g_v, g_d = c["ref_rmse"] - ens_val["cnn_lstm_kan"]["vmax"], c["ref_dvmax"] - c["e_kan"].rmse
    out += ["### 4.3 Ensemble das 5 seeds (é outro modelo, 5× mais caro)", "",
            tabela(["Variante", "Vmax RMSE: único", "ensemble", "ganho", "ΔVmax RMSE: único", "ensemble", "ganho", "RI AUC (ensemble)", "RI AP (ensemble)"], linhas, "lrrrrrrrr"), "",
            f"Na proposta, o ganho do ensemble é de {num(g_v)} nó em Vmax e de {num(g_d)} nó em ΔVmax ({'maior' if g_d > g_v else 'menor'} em ΔVmax, o que sugere variância mais alta na cabeça de ΔVmax). "
            "AUC e AP do ensemble usam os 5 folds juntos.", "",
            f"*Convenção (a da análise independente): o RMSE de Vmax do ensemble usa as {inteiro(n['janelas'])} janelas juntas e o de ΔVmax é a média dos RMSE dos 5 folds, como nas tabelas do notebook; o «modelo único» é a média das 25 execuções. "
            f"Na outra convenção os valores do ensemble da proposta seriam {num(ens_val['cnn_lstm_kan']['vmax_media_folds'], 3)} (Vmax, média dos folds) e {num(ens_val['cnn_lstm_kan']['dvmax_agrupado'], 3)} (ΔVmax, janelas juntas), "
            f"contra {num(ens_val['cnn_lstm_kan']['vmax'], 3)} e {num(ens_val['cnn_lstm_kan']['dvmax'], 3)}: diferenças de {num(abs(ens_val['cnn_lstm_kan']['vmax'] - ens_val['cnn_lstm_kan']['vmax_media_folds']), 3)} e "
            f"{num(abs(ens_val['cnn_lstm_kan']['dvmax'] - ens_val['cnn_lstm_kan']['dvmax_agrupado']), 3)} nó, pequenas diante dos ganhos.*", ""]

    # 4.4 comparações estatísticas
    b = boot.merge(fold[["comparacao", "tarefa", "metrica", "folds_ref_melhor"]], on=["comparacao", "tarefa", "metrica"], how="left")
    nd = lambda x: 3 if abs(x) < 0.05 else 2                                            # noqa: E731
    linhas = [[COMPARACOES[r.comparacao], TAREFA[r.tarefa], r.metrica.upper(), f"{num(r.dif, nd(r.dif), True)} {ic(r.ic95_inf, r.ic95_sup, nd(r.dif))}",
               pv(r.p_holm), VEREDITO[r.veredito], r.folds_ref_melhor] for r in b.itertuples()]
    lv = linha_da(b, "cnn_lstm_kan − cnn_kan_1frame", "vmax", "rmse")
    ld = linha_da(b, "cnn_lstm_kan − cnn_kan_1frame", "dvmax", "rmse")
    kd, kde = linha_da(b, "cnn_lstm_kan − cnn_lstm_mlp", "dvmax", "rmse"), linha_da(d["boot_ens"], "cnn_lstm_kan − cnn_lstm_mlp", "dvmax", "rmse")
    mae_h = linha_da(b, "cnn_lstm_kan − baseline_historico", "dvmax", "mae")
    w = c["wexec"]("cnn_kan_1frame", "rmse_vmax")
    out += ["### 4.4 Comparações estatísticas (bootstrap por ciclone, modelo único)", "",
            tabela(["Comparação", "Tarefa", "Métrica", "Diferença (proposta − outra) [IC 95%]", "p (Holm)", "Veredito", "Folds em que a proposta erra menos"], linhas, "lllrrll"), "",
            f"O menor p possível com {inteiro(cfg['bootstrap_reps'])} réplicas é ≈ 0,001 e, com 12 testes, **{num(b.p_holm.min(), 3)} é o piso do p ajustado**. Leitura:", "",
            f"1. **A LSTM ajuda (robusto).** Vmax: {num(lv.dif, sinal=True)} nó de RMSE ({num(100 * lv.dif / c['m1']['cnn_kan_1frame'], 0, True)}%), IC que exclui 0, proposta melhor em {lv.folds_ref_melhor} folds e em {w.ref_melhor_em} execuções; "
            f"ΔVmax: {num(ld.dif, sinal=True)} nó, em {ld.folds_ref_melhor} folds. A variante de 1 quadro, que ainda recebe DTA/SDTA, é um baseline forte («estilo TCC com 4 canais»): o ganho vem da sequência.",
            f"2. **KAN × MLP é empate em Vmax e frágil em ΔVmax.** Em ΔVmax o modelo único dá {num(kd.dif, sinal=True)} nó (p Holm {pv(kd.p_holm)}), mas em {kd.folds_ref_melhor} folds; com o ensemble das seeds o efeito cai para "
            f"{num(kde.dif, 3, True)} {ic(kde.ic95_inf, kde.ic95_sup, 3)} e deixa de ser significativo (p Holm {pv(kde.p_holm)}). As cabeças são só ≈ {num(100 * 2 * n['par']['kan'] / n['par']['total'], 0)}% dos parâmetros: esta ablação tem pouca alavanca.",
            "3. **A multitarefa piora levemente o Vmax**: a variante só-Vmax é melhor nos 5 folds. A premissa de que a multitarefa regulariza e melhora o Vmax **não se confirmou** com o peso 0,5 para ΔVmax.",
            f"4. **ΔVmax × baseline de histórico**: o modelo único **empata** (o IC do RMSE contém 0; no MAE o p ajustado é {pv(mae_h.p_holm)}).", ""]

    # 4.5 ΔVmax e RI
    dvb = d["dvb"].set_index("modelo")
    linhas = [[ROTULO[k]] + [virgula(dvb.loc[k, col]) for col in ("rmse_dvmax", "mae_dvmax", "pod_ri", "far_ri", "csi_ri", "limiar_ri_aj", "pod_ri_aj", "far_ri_aj", "csi_ri_aj", "auc_ri", "ap_ri")]
              for k in ("cnn_lstm_kan", "cnn_lstm_mlp", "cnn_kan_1frame", "baseline_historico", "baseline_zero", "baseline_media")]
    out += ["### 4.5 ΔVmax e intensificação rápida (RI) contra os baselines", "",
            tabela(["Modelo", "ΔVmax RMSE", "ΔVmax MAE", "POD (30)", "FAR (30)", "CSI (30)", "Limiar ajustado (nós)", "POD aj.", "FAR aj.", "CSI aj.", "AUC", "AP"], linhas, "lrrrrrrrrrrr"), "",
            f"Média ± desvio (25 execuções para as redes; 5 folds para os baselines). «(30)» = alarme quando o ΔVmax previsto é ≥ 30 nós; «aj.» = limiar escolhido na validação interna. "
            f"O limiar fixo de 30 nós é enganoso: a regressão por mínimos quadrados encolhe as previsões em direção à média (RI é só {pct(c['prev'])} das amostras), então a rede quase nunca chega a 30 e o POD sai baixo mesmo quando ela ordena bem os casos; "
            "por isso AUC e AP são as medidas justas.", ""]
    r_ = d["ri"].set_index("escore")
    linhas = []
    for k, rot in (("cnn_lstm_kan (ensemble)", "`cnn_lstm_kan` (ensemble)"), ("cnn_lstm_mlp (ensemble)", "`cnn_lstm_mlp` (ensemble)"), ("cnn_kan_1frame (ensemble)", "`cnn_kan_1frame` (ensemble)"),
                   ("baseline_historico", "baseline de histórico"), ("histórico + rede (ensemble)", "histórico + rede (empilhamento; exploratório)")):
        x = r_.loc[k]
        sem = np.isnan(x.dif_auc_vs_ref)
        linhas.append([rot, f"{num(x.auc, 3)} {ic_abs(x.auc_ic_inf, x.auc_ic_sup)}", f"{num(x.ap, 3)} {ic_abs(x.ap_ic_inf, x.ap_ic_sup)}",
                       "—" if sem else f"{num(x.dif_auc_vs_ref, 3, True)} {ic(x.dif_auc_ic_inf, x.dif_auc_ic_sup, 3)}",
                       "—" if sem else f"{num(x.dif_ap_vs_ref, 3, True)} {ic(x.dif_ap_ic_inf, x.dif_ap_ic_sup, 3)}",
                       "—" if sem else pv(x.p_auc_holm), "—" if sem else pv(x.p_ap_holm)])
    u = ens.loc["cnn_lstm_kan" + UNI]
    um, hh = r_.loc["cnn_kan_1frame (ensemble)"], r_.loc["baseline_historico"]
    out += ["**Ranking de RI sem limiar, com os 5 folds juntos e IC por ciclone (ensemble das seeds).** A diferença é «proposta (ensemble) − outra»; positiva = a proposta ordena melhor.", "",
            tabela(["Escore", "AUC [IC 95%]", "AP [IC 95%]", "Δ AUC vs proposta", "Δ AP vs proposta", "p AUC (Holm)", "p AP (Holm)"], linhas, "lrrrrrr"), "",
            f"Prevalência de RI: {pct(c['prev'])} (um AP sem habilidade ≈ {num(c['prev'] / 100, 3)}). O ensemble **supera o histórico** em AUC e em AP (p ajustado {pv(hh.p_auc_holm)}); "
            f"o modelo único também fica acima (AUC {num(u.auc, 3)}, AP {num(u.ap, 3)}, com os 5 folds juntos). "
            f"A **sequência quase não ajuda a RI** (LSTM × 1 quadro: AUC {num(um.dif_auc_vs_ref, 3, True)}, p ajustado {pv(um.p_auc_holm)}: não significativo; AP {num(um.dif_ap_vs_ref, 3, True)}, p ajustado {pv(um.p_ap_holm)}): "
            "o ganho da sequência está no Vmax, não na RI; KAN × MLP: nada.", ""]
    pu, pe = ens.loc["histórico + cnn_lstm_kan — modelo único (média das seeds)"], ens.loc["histórico + cnn_lstm_kan — ensemble"]
    out += [f"**Imagens e histórico se complementam (exploratório).** Uma regressão linear que junta as variáveis do histórico e a previsão de ΔVmax da rede (fora da amostra) dá RMSE de ΔVmax **{num(pu.rmse)}** (modelo único) e **{num(pe.rmse)}** (ensemble) "
            f"contra {num(c['hist_rmse'])} do histórico sozinho ({num(pu.rmse - c['hist_rmse'], sinal=True)} a {num(pe.rmse - c['hist_rmse'], sinal=True)} nó), AUC {num(pu.auc, 3)} / {num(pe.auc, 3)} e AP {num(pu.ap, 3)} / {num(pe.ap, 3)}. "
            "Ressalva: o empilhamento é ajustado nos outros folds com previsões de modelos que viram o fold avaliado (vazamento leve) e **não substitui treinar a rede com o histórico como entrada**.", ""]

    # 4.6 erro por intensidade
    it = d["inten"]
    a1, um1 = it[it.variante == "cnn_lstm_kan"].reset_index(drop=True), it[it.variante == "cnn_kan_1frame"].reset_index(drop=True)
    linhas = [[a1.faixa[i], inteiro(a1.n[i]), num(a1.rmse[i]), num(a1.mae[i]), num(a1.vies[i], sinal=True), num(um1.rmse[i])] for i in range(len(a1))]
    out += ["### 4.6 Erro por faixa de intensidade (regressão à média)", "",
            tabela(["Faixa de Vmax real (nós)", "Janelas", "RMSE", "MAE", "Viés (previsto − real)", "RMSE da variante de 1 quadro"], linhas, "lrrrrr"), "",
            "Previsão da `cnn_lstm_kan` (média das seeds) sobre todas as janelas. O viés muda de sinal com a intensidade: **superestima as depressões** "
            f"({num(a1.vies.iloc[0], 1, True)} nós) e **subestima os ciclones fortes** ({num(a1.vies.iloc[-1], 1, True)} nós na faixa mais alta), o padrão de quem regride à média; o RMSE global esconde isso. "
            f"A variante de 1 quadro é pior em {'todas as ' + str(len(a1)) if int((um1.rmse > a1.rmse).sum()) == len(a1) else str(int((um1.rmse > a1.rmse).sum())) + ' das ' + str(len(a1))} faixas.", "",
            figura("04_erro_por_intensidade.png", "MAE de Vmax por faixa de intensidade, seed 0 (figura 04 do notebook; faixas agrupadas).", 560), ""]

    # 4.7 2017
    r17, n17 = d["r2017"], nb["nb2017"].get("cnn_lstm_kan")
    linhas = [[ROTULO.get(m, m), num(x.rmse_2017), num(x.mae_2017), inteiro(x.n_janelas_2017),
               f"5 seeds (dp do RMSE entre seeds: {num(x.rmse_dp_seeds)})" if not np.isnan(x.rmse_dp_seeds) else "1 treino (TCC)"] for m, x in zip(r17.modelo, r17.itertuples())]
    f17 = r17.set_index("modelo")
    out += ["### 4.7 Só as janelas de 2017, ao lado do TCC", "",
            tabela(["Modelo", "RMSE", "MAE", "Janelas", "Observação"], linhas, "lrrrl"), "",
            "**Comparação não estrita:** (i) aqui as janelas de 2017 vêm da validação cruzada por ciclone, então os modelos que as preveem treinaram com ciclones de outros anos e de 2017; o TCC treinou em 2003–2014 e testou em 2017; "
            f"(ii) são {inteiro(f17.loc['cnn_lstm_kan', 'n_janelas_2017'])} janelas contra {inteiro(f17.loc['CNN-TC (TCC)', 'n_janelas_2017'])} imagens; (iii) a entrada é outra (sequência de 5 quadros com IR, PMW, DTA e SDTA, contra uma imagem isolada). "
            f"A variante de 1 quadro ({num(f17.loc['cnn_kan_1frame', 'rmse_2017'])}) fica no nível do CNN-TC ({num(f17.loc['CNN-TC (TCC)', 'rmse_2017'])}): o encoder é compatível com o do TCC e o ganho vem da sequência. "
            + (f"A tabela de 2017 do próprio notebook usa só a seed 0 (RMSE {num(n17['rmse'])} para a `cnn_lstm_kan`); a média das 5 seeds, {num(f17.loc['cnn_lstm_kan', 'rmse_2017'])}, é o número a citar. " if n17 else "")
            + f"O teste de 2017 é mais fácil que a média da validação cruzada (RMSE {num(c['ref_rmse'])}).", ""]

    # 4.8 faixa horizontal
    fx, lim, sens = d["ana"]["faixa"], d["faixa_lim"], d["sens"].set_index("variante")
    ch = d["faixa_cic"]
    resto = ch[ch.id_ciclone != "201407E"]
    bs = boot.merge(d["boot_sem"], on=["comparacao", "tarefa", "metrica"], suffixes=("", "_sem"))
    igual = bool(len(bs) == len(boot) and (bs.veredito == bs.veredito_sem).all())
    p5, ad = fx["perto_5graus"], fx["antes_depois_vmax100"]
    out += ["### 4.8 A faixa horizontal do gráfico predito × real: um defeito dos dados", "",
            f"No gráfico predito × real há uma faixa de janelas com Vmax real entre ≈ 40 e 140 nós previstas em ≈ 25 nós **pelas quatro variantes**: isso aponta para a entrada ou o rótulo, não para a arquitetura. "
            f"Contando as janelas com Vmax real ≥ {int(lim.vmax_real_min[0])} e previsto ≤ {int(lim.previsto_max[0])} nós: **{lim.n[0]}** (de {lim.ciclones[0]} ciclones), das quais **{lim.n_201407E[0]}** são do ciclone `201407E`; "
            f"com Vmax real ≥ {int(lim.vmax_real_min[1])} são {lim.n_201407E[1]} de {lim.n[1]}, com ≥ {int(lim.vmax_real_min[2])}, {lim.n_201407E[2]} de {lim.n[2]} e com ≥ {int(lim.vmax_real_min[3])}, {lim.n_201407E[3]} de {lim.n[3]}. "
            f"As outras {int(resto.n.sum())} são ciclones de Vmax {int(resto.vmax_min.min())}–{int(resto.vmax_max.max())} previstos em ≈ {num(np.average(resto.pred_media, weights=resto.n), 0)} nós (a cauda comum da regressão à média).", "",
            "**Causa.** A longitude do `201407E` no TCIR nunca passa de −179,8°: depois da linha de data ela volta a *crescer* (−179,3°, −178,7°…) quando deveria continuar para oeste, isto é, está gravada com o sinal trocado. "
            "O recorte de 201×201 px é centrado nessa posição, que passa a ficar **2·(180 − |lon|)° a leste do ciclone**: por volta de 07/ago/2014 15 h o ciclone sai do recorte 92×92 que a rede usa e, em 08/ago 00 h, do quadro inteiro. "
            f"Entre as janelas do `201407E` com Vmax real ≥ 100 nós, as {ad['n_antes']} anteriores à virada têm erro absoluto médio de {num(ad['mae_antes'], 1)} nós e as {ad['n_depois']} posteriores, de {num(ad['mae_depois'], 1)} nós "
            f"({ad['depois_previsto_ate_35']} delas previstas em ≤ 35 nós).", "",
            figura("fig_201407E_quadros_longitude_trocada.png", "Quadros de IR do `201407E`: «+» = centro do recorte (posição gravada), retângulo ciano = recorte 92×92 que a rede usa, «×» = onde o ciclone estaria se a longitude tivesse o sinal trocado. "
                   "O ciclone aparece exatamente onde a hipótese manda e sai do recorte."), "",
            figura("fig_faixa_antes_depois.png", f"`cnn_lstm_kan` (média das seeds): com todas as janelas (esquerda, RMSE {num(fx['rmse_todas'])}) e sem as {fx['janelas_afetadas']} janelas afetadas do `201407E` (direita, RMSE {num(fx['rmse_sem_afetadas'])}); em vermelho, as janelas afetadas."), "",
            f"**Não é um problema geral da linha de data.** Os outros {p5['ciclones'] - 1} ciclones com janelas a ≤ 5° da linha ({inteiro(p5['n_outros'])} janelas) têm RMSE {num(p5['rmse_outros_ciclones'])} (resto do conjunto: {num(p5['rmse_resto_do_conjunto'])}), "
            f"{p5['faixa_outros']} janelas na faixa e {p5['erro_grande_outros']} com erro ≥ {int(cfg['outlier_erro_nos'])} nós.", "",
            f"**Peso e sensibilidade.** As **{fx['janelas_afetadas']} janelas** ({pct(100 * fx['janelas_afetadas'] / n['janelas'], 2)} do total) respondem por **{pct(fx['pct_sq_afetadas'])} do erro quadrático**. "
            f"Sem elas o RMSE de Vmax cai de {num(sens.loc['cnn_lstm_kan', 'rmse_vmax'])} para {num(sens.loc['cnn_lstm_kan', 'rmse_vmax_sem_afetadas'])} (modelo único; {num(sens.loc['cnn_lstm_kan', 'dif'], 2, True)} nó, parecido nas quatro variantes) "
            f"e o do ensemble de {num(fx['rmse_todas'])} para {num(fx['rmse_sem_afetadas'])}; o RMSE de ΔVmax praticamente não muda ({num(sens.loc['cnn_lstm_kan', 'rmse_dvmax'])} → {num(sens.loc['cnn_lstm_kan', 'rmse_dvmax_sem_afetadas'])}). "
            f"Refazendo os {len(boot)} testes por ciclone sem elas, **{'os vereditos são os mesmos' if igual else 'ALGUM VEREDITO MUDA (conferir)'}**.", "",
            f"**Os demais erros grandes não são de recorte.** Dos {fx['erro_grande_n']} erros com valor absoluto ≥ {int(cfg['outlier_erro_nos'])} nós ({fx['erro_grande_ciclones']} ciclones; {pct(d['ana']['log']['erro_grande_log']['pct_sq'])} do erro quadrático), "
            f"{fx['erro_grande_afetadas']} são do `201407E`; o resto são ciclones intensos subestimados em 40–58 nós, com o ciclone centrado e bem formado nos quadros vistos: regressão à média ou erro de rótulo.", "",
            "**Não verificado:** a causa foi concluída pelos quadros e pela trilha, **não contra a longitude original** (GridSat/best track). O TCC, com o mesmo dataset, provavelmente carrega as mesmas imagens (não conferido). "
            "Uma triagem pela trilha só acusou outro ciclone (`200310E`, inconclusivo, sem peso no erro); um ciclone fraco com o recorte errado seria previsto como fraco e não apareceria nos erros, então outros casos não estão excluídos.", ""]

    # 4.9 sobreajuste
    so = d["sobre"].set_index("variante")
    lstm_v = [k for k in VARIANTES if k != "cnn_kan_1frame"]
    linhas = [[ROTULO[k], f"{int(so.loc[k, 'pararam_antes'])}/{int(so.loc[k, 'execucoes'])}", num(so.loc[k, "melhor_epoca_media"], 1)]
              + [num(so.loc[k, col]) for col in ("val_rmse_epoca_10", "val_rmse_epoca_15", "val_rmse_epoca_20", "val_rmse_ultima", "val_rmse_no_melhor")] for k in VARIANTES]
    sv = so.loc[SEM_DVMAX]
    folga = (so.val_rmse_ultima - so.val_rmse_no_melhor)[lstm_v]
    out += ["### 4.9 Sobreajuste e early stopping", "",
            tabela(["Variante", "Pararam antes da época 25", "Melhor época (média)", "RMSE de validação na época 10", "15", "20", "25 (última)", "no melhor ponto"], linhas, "lrrrrrrr"), "",
            f"- **{int(so.loc[lstm_v, 'pararam_antes'].sum())} das {int(so.loc[lstm_v, 'execucoes'].sum())} execuções com LSTM pararam antes da época {cfg['epocas_max']}** (a de 1 quadro nunca parou); "
            f"na média, o RMSE da última época fica só {num(folga.min()) if num(folga.min()) == num(folga.max()) else num(folga.min()) + ' a ' + num(folga.max())} nó acima do melhor: não há degradação forte no fim.",
            f"- **Há folga entre treino e validação.** Na variante só-Vmax a perda de treino é o MSE de Vmax padronizado, então o RMSE de treino é ≈ **{num(sv.rmse_treino_aprox_ultima, 1)} nós** (medido com *dropout* e rotação, logo é um limite superior do erro de treino em modo de avaliação) "
            f"contra **{num(sv.rmse_val_ultima_so_vmax, 1)} nós** na validação: folga de ≥ {num(sv.gap_val_menos_treino, 1)} nós (o erro de treino é ≥ {num(100 * sv.gap_val_menos_treino / sv.rmse_val_ultima_so_vmax, 0)}% menor).",
            f"- O ganho do ensemble (seção 4.3) confirma **variância alta**, {'maior' if g_d > g_v else 'menor'} na cabeça de ΔVmax. O notebook completo não grava o RMSE de treino em modo de avaliação, então a folga só se estima de forma indireta. "
            "Uma medida direta existe em **outra rodada** (o notebook rápido de 1 seed, 2026-10-23; notas do projeto, seção 9.11): folga de +3,6 nós em Vmax e de +9,0 nós em ΔVmax, isto é, a cabeça de ΔVmax decora o treino.", ""]

    # 4.10 explicabilidade
    kan, lat = d["ana"]["log"]["kan"], d["ana"]["latentes"]
    out += ["### 4.10 Explicabilidade", "",
            f"**KAN** (`cnn_lstm_kan`, 1º fold executado). As 10 variáveis latentes mais importantes concentram **{pct(kan['vmax']['top10_pct'])}** (Vmax) e **{pct(kan['dvmax']['top10_pct'])}** (ΔVmax) da importância "
            f"(se fosse uniforme: {pct(kan['vmax']['uniforme_pct'])}); o número efetivo de variáveis é **{kan['vmax']['n_efetivo']} de {kan['vmax']['n_total']}**: **não há esparsidade**. "
            f"Das 10 mais importantes de cada tarefa, {kan['sobreposicao']['n']} coincidem ({', '.join('h' + str(x) for x in kan['sobreposicao']['comuns'])}) contra {num(kan['sobreposicao']['esperado_acaso'], 1)} esperado ao acaso "
            f"(P = {num(kan['sobreposicao']['p'], 2)}; teste isolado, sem correção). As splines da 1ª camada de Vmax são **suaves e quase monótonas** (h113 e h121 decrescentes; h17 e h25 crescentes): a cabeça se comporta como um mapa quase aditivo e monótono, coerente com KAN ≈ MLP.", "",
            figura("07_kan_importancia_vmax.png", "As 20 variáveis latentes de h(t) mais importantes para a cabeça de Vmax (figura 07 do notebook).", 620), "",
            figura("08_kan_splines_vmax.png", "Splines aprendidas pela cabeça KAN de Vmax: linha de cima, φ das 4 variáveis principais; linha de baixo, a função da 2ª camada (cinza: distribuição dos valores; figura 08 do notebook)."), "",
            f"**O que as variáveis latentes codificam** (Spearman). A {lat['Vmax real']['variavel']} correlaciona {num(lat['Vmax real']['rho'])} com o Vmax real; a maior correlação com o **ΔVmax real** é só {num(abs(lat['ΔVmax real']['rho']))} ({lat['ΔVmax real']['variavel']}): "
            "a cabeça de ΔVmax se apoia em variáveis que codificam a **intensidade atual**, como o baseline de histórico. As variáveis não têm nome físico (uma h_i é uma coordenada arbitrária do espaço da LSTM); interpreta-se a forma da curva, não um significado meteorológico.", "",
            f"**Grad-CAM.** Os mapas são difusos (grade 4×4 ampliada); nos exemplos o calor cai em manchas de convecção, nem sempre no núcleo (no exemplo de 120 nós, o olho visível em `t` não é destacado), "
            f"e o mapa médio sobre {cfg['gradcam_amostras']} janelas se concentra no quadrante inferior direito e, em `t`, no centro-direita, não no olho. **Sem teste de sanidade** (randomização dos pesos, inserção/remoção de pixels): "
            "não usar como evidência de «a rede olha o olho».", "",
            figura("05_gradcam_exemplos_vmax.png", "Grad-CAM do encoder para três janelas (Vmax real ≈ 30, 70 e 120 nós), um mapa por quadro (figura 05 do notebook)."), "",
            figura("06_gradcam_medio_vmax.png", f"Grad-CAM médio sobre {cfg['gradcam_amostras']} janelas de teste (figura 06 do notebook).", 760), ""]
    return "\n".join(out)


def secao_sustenta(d: dict, b: pd.DataFrame, bens: pd.DataFrame, c: dict) -> str:
    r = lambda comp, tar, mt, tab=b: linha_da(tab, comp, tar, mt)                       # noqa: E731
    lstm_v, lstm_d = r("cnn_lstm_kan − cnn_kan_1frame", "vmax", "rmse"), r("cnn_lstm_kan − cnn_kan_1frame", "dvmax", "rmse")
    kan_v, kan_d, kan_de = r("cnn_lstm_kan − cnn_lstm_mlp", "vmax", "rmse"), r("cnn_lstm_kan − cnn_lstm_mlp", "dvmax", "rmse"), r("cnn_lstm_kan − cnn_lstm_mlp", "dvmax", "rmse", bens)
    mt = r("cnn_lstm_kan − cnn_lstm_kan_vmax", "vmax", "rmse")
    h, he = r("cnn_lstm_kan − baseline_historico", "dvmax", "rmse"), r("cnn_lstm_kan − baseline_historico", "dvmax", "rmse", bens)
    ri, fo, kan = d["ri"].set_index("escore"), c["fold"], d["ana"]["log"]["kan"]
    pexec = c["wexec"]("cnn_lstm_mlp", "rmse_dvmax")
    inten = d["inten"][d["inten"].variante == "cnn_lstm_kan"].reset_index(drop=True)
    linhas = [
        ("A LSTM melhora a estimativa de Vmax", "✔ sustentada (robusta)", f"{num(lstm_v.dif, sinal=True)} nó {ic(lstm_v.ic95_inf, lstm_v.ic95_sup)}, {fo('cnn_lstm_kan − cnn_kan_1frame', 'vmax', 'rmse').folds_ref_melhor} folds"),
        ("A sequência melhora ΔVmax", "✔ sustentada (efeito pequeno)", f"{num(lstm_d.dif, sinal=True)} nó {ic(lstm_d.ic95_inf, lstm_d.ic95_sup)}, {fo('cnn_lstm_kan − cnn_kan_1frame', 'dvmax', 'rmse').folds_ref_melhor} folds"),
        ("A KAN é melhor que o MLP em Vmax", "✘ **não** sustentada (empate)", f"{num(kan_v.dif, 3, True)} {ic(kan_v.ic95_inf, kan_v.ic95_sup, 3)}; p Holm {pv(kan_v.p_holm)}"),
        ("A KAN é melhor que o MLP em ΔVmax", "✘ frágil: não afirmar",
         f"modelo único {num(kan_d.dif, 3, True)} (p Holm {pv(kan_d.p_holm)}), {fo('cnn_lstm_kan − cnn_lstm_mlp', 'dvmax', 'rmse').folds_ref_melhor} folds; ensemble {num(kan_de.dif, 3, True)} (p Holm {pv(kan_de.p_holm)}). "
         f"O teste por execução (exploratório) dá p Holm {pv(pexec.p_holm)}" + (": artefato da pseudorreplicação (as 25 execuções valem ≈ 5 observações)" if pexec.p_holm < d["cfg"]["alfa"] else "")),
        ("A multitarefa melhora o Vmax", "✘ **falsa**: piora", f"{num(mt.dif, sinal=True)} nó {ic(mt.ic95_inf, mt.ic95_sup)}; a só-Vmax é melhor em {5 - int(fo('cnn_lstm_kan − cnn_lstm_kan_vmax', 'vmax', 'rmse').folds_ref_melhor.split('/')[0])}/5 folds"),
        ("A rede (modelo único) supera o baseline de histórico em ΔVmax", "✘ empate", f"{num(h.dif, sinal=True)} {ic(h.ic95_inf, h.ic95_sup)}"),
        ("O ensemble de 5 seeds supera o baseline de histórico em ΔVmax", "✔ sustentada", f"{num(he.dif, sinal=True)} {ic(he.ic95_inf, he.ic95_sup)}; custa 5× mais"),
        ("A rede ordena a RI melhor que o histórico (AUC/AP)", "✔ sustentada (ensemble; o modelo único também fica acima)",
         f"AUC {num(ri.loc['baseline_historico', 'dif_auc_vs_ref'], 3, True)}, AP {num(ri.loc['baseline_historico', 'dif_ap_vs_ref'], 3, True)} (p Holm {pv(ri.loc['baseline_historico', 'p_ap_holm'])})"),
        ("A sequência melhora a RI", "✘ não significativa", f"AUC {num(ri.loc['cnn_kan_1frame (ensemble)', 'dif_auc_vs_ref'], 3, True)} (p Holm {pv(ri.loc['cnn_kan_1frame (ensemble)', 'p_auc_holm'])})"),
        ("Imagens e histórico se complementam", "~ indicativo", "empilhamento exploratório, com vazamento leve (seção 4.5)"),
        ("A KAN é esparsa e interpretável", "✘ não (sem esparsificar)", f"número efetivo de variáveis {kan['vmax']['n_efetivo']} de {kan['vmax']['n_total']}; splines quase monótonas"),
        ("O Grad-CAM mostra que a rede olha o olho", "✘ não demonstrado", "mapas difusos e sem teste de sanidade"),
        ("Supera o TCC em 2017", "~ indicativo", "comparação não estrita (seção 4.7)"),
    ]
    return "## 5. O que os resultados sustentam e o que não\n\n" + tabela(["Afirmação", "Situação", "Base"], linhas, "lll") + "\n" + \
        f"\n*Diferença = proposta − outra; negativa = a proposta erra menos. Viés da proposta na faixa mais alta de intensidade: {num(inten.vies.iloc[-1], 1, True)} nós (seção 4.6).*\n"


def secao_limites(d: dict, n: dict, c: dict) -> str:
    inten = d["inten"][d["inten"].variante == "cnn_lstm_kan"].reset_index(drop=True)
    return f"""## 6. Limitações

- **Uma execução** ({n['n_exec']} treinos, uma só rodada no Kaggle): as seeds medem o ruído do treino, não a variação entre rodadas. Os intervalos do bootstrap embutem a variação entre ciclones.
- **Modelo único × ensemble** (seção 4.3): as comparações do protocolo são de modelo único; o ensemble muda o veredito de ΔVmax × histórico.
- **O baseline de histórico usa o Vmax real** de t, t−3 h, t−6 h e t−12 h, informação que a rede não recebe. Dar o histórico como entrada da rede muda o modelo proposto e exige nova execução completa (decisão em aberto).
- **Comparação com o TCC não é estrita** (seção 4.7).
- **Horizonte fixo e discreto.** A LSTM supõe passos de 3 h e janelas com buracos são descartadas; um instante sem imagem não teria previsão. Neural ODE (tempo contínuo) é o caminho natural.
- **RI é um evento raro** ({pct(c['prev'])}) e a cabeça é de regressão: a rede ordena bem, mas o alarme com limiar fixo falha. Cabeça de classificação (BCE ou *focal loss*) fica como trabalho futuro.
- **Regressão à média** nas categorias altas (viés de {num(inten.vies.iloc[-1], 1, True)} nós na faixa mais alta) e **sobreajuste moderado**, maior na cabeça de ΔVmax.
- **KAN sem esparsificação nem grid adaptativo**: a importância se espalha entre as variáveis latentes; as splines mostram a forma da relação, não um significado meteorológico.
- **Grad-CAM grosseiro** (4×4) e sem teste de sanidade.
- **Defeito de dados no TCIR** (`201407E`; seção 4.8): {d['ana']['faixa']['janelas_afetadas']} janelas com o recorte fora do ciclone; outros casos mascarados não estão excluídos.
- **AP com empates:** a função de precisão média ordena escores iguais pela ordem das janelas; as previsões das redes não têm empates, mas o AP do baseline de histórico pode ser afetado (efeito não medido; exige o `data/` completo).
- **Rótulos do *best track*** têm incerteza própria (cerca de 80% dos Vmax do TCIR são múltiplos de 5 nós, verificado nos dados), o que limita o RMSE alcançável.
"""


def secao_verificacao(v: dict) -> str:
    sci = lambda x: f"{x:.2e}".replace(".", ",")                                       # noqa: E731  (vírgula decimal também aqui)
    linhas = [(c, sci(x)) for c, x in v["dif_por_metrica"].items()] + [("ensemble das 5 seeds (RMSE de Vmax com as janelas juntas; RMSE e MAE de ΔVmax como média dos 5 folds)", sci(v["dif_ensemble"]))]
    return f"""## 7. Verificação independente (feita por este script)

Antes de escrever este documento, o `montar_resultado.py` releu as **{v['execucoes']} predições** (`runs/*.npz`, sem pickle) e recalculou as métricas. Diferença máxima, em valor absoluto, contra as tabelas (tolerância {v['tolerancia']:.0e}):

{tabela(['Quantidade', 'Diferença máxima'], linhas, 'lr')}

- Cada janela é testada exatamente uma vez por (variante, seed) ({inteiro(v['n_janelas'])} janelas): **{'confere' if v['particao_ok'] else 'NÃO CONFERE'}**.
- As 5 seeds de um fold testam as mesmas janelas, na mesma ordem (pré-condição do ensemble): **confere** (o script aborta se não).
- Resultado: **{'tudo confere' if v['ok'] else 'DIVERGÊNCIA'}**. A análise de 2026-10-16 (`analisar_resultados_pipeline.py`) já havia conferido as janelas contra o conjunto reconstruído dos dados; isso não foi refeito aqui porque o `data/` local não tem o arquivo `TCIR-ATLN_EPAC_WPAC-001.h5` completo. \
Os números de 2017, do bootstrap, de AUC/AP com IC, do empilhamento e da sensibilidade vêm daquela análise (`resultados_analise_pipeline/`).
"""


def secao_reproduzir() -> str:
    return """## 8. Como reproduzir

```bash
python3 -I scripts/analise/analisar_resultados_pipeline.py   # análise independente (≈ 1,5 min; exige data/ completo) → resultados_analise_pipeline/
python3 -I scripts/relatorio/figura_metodologia.py           # figura da metodologia → docs/resultado_figuras/metodologia_pipeline.png (.pdf)
python3 -I scripts/relatorio/montar_resultado.py             # este documento (reconfere as 100 predições antes de escrever)
```

| Arquivo | O que é |
|---|---|
| `resultados_pipeline/saida_pipeline/` | saída do notebook no Kaggle: `config.json`, `resumo_final.json`, `tabelas/`, `figuras/`, `runs/` (100 `.npz` e 20 `.pt`) |
| `resultados_pipeline/__notebook__.ipynb` | o notebook executado, com o log (dado não confiável: ler só como JSON, nunca executar) |
| `resultados_analise_pipeline/` | relatório, tabelas e figuras da análise independente |
| `docs/resultado_figuras/` | as figuras usadas neste documento |
| `scripts/relatorio/figura_metodologia.py`, `scripts/relatorio/montar_resultado.py` | geram a figura da metodologia e este documento |
"""


# ----------------------------------------------------------------------------- principal
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--saida-pipeline", type=Path, default=RAIZ / "resultados_pipeline" / "saida_pipeline")
    ap.add_argument("--analise", type=Path, default=RAIZ / "resultados_analise_pipeline")
    ap.add_argument("--md", type=Path, default=RAIZ / "docs" / "resultado--.md")
    ap.add_argument("--figuras", type=Path, default=RAIZ / "docs" / "resultado_figuras")
    ap.add_argument("--zip-baixado", type=Path, default=None, help="o resultados_pipeline.zip que o usuário baixou (confere o SHA-256)")
    ap.add_argument("--notebook-baixado", type=Path, default=None, help="o notebook .ipynb que o usuário baixou (confere célula a célula; só lê como JSON)")
    a = ap.parse_args()

    d = carregar(a.saida_pipeline, a.analise)
    n = carregar_numeros(a.saida_pipeline, a.analise)
    ver = verificar_npz(a.saida_pipeline, d["met"], d["resumo"], d["ens"], n["janelas"])
    print("verificação:", json.dumps({k: v for k, v in ver.items() if k != "rmse_ensemble"}, default=float))
    if not ver["ok"]:
        print("[erro] as predições não batem com as tabelas: o documento NÃO foi escrito", file=sys.stderr)
        return 1
    nb = ler_notebook(a.saida_pipeline.parent / "__notebook__.ipynb", RAIZ, a.zip_baixado, a.notebook_baixado)

    a.figuras.mkdir(parents=True, exist_ok=True)
    for orig, nome in FIGURAS:
        shutil.copy2((a.saida_pipeline / "figuras" if orig == "saida" else a.analise / "figuras") / nome, a.figuras / nome)
    if not (a.figuras / "metodologia_pipeline.png").exists():
        print("[erro] falta docs/resultado_figuras/metodologia_pipeline.png: rode `python3 -I scripts/relatorio/figura_metodologia.py` antes", file=sys.stderr)
        return 1

    c = valores_comuns(d, n, ver["rmse_ensemble"])
    b, bens = d["boot"], d["boot_ens"]
    cabeca = f"""# Resultados do experimento

**CNN encoder (pesos compartilhados) → LSTM → cabeças KAN multitarefa para estimar a intensidade (Vmax) e a variação em {n['cfg']['horizonte_horas']} h (ΔVmax) de ciclones tropicais a partir de sequências de imagens de satélite (TCIR).**

> Gerado por `montar_resultado.py` em {DATA_DOC} a partir dos arquivos que o notebook gravou (`config.json`, `resumo_final.json`, `tabelas/`, `runs/`) e da análise independente `analisar_resultados_pipeline.py`. \
Os números vêm desses arquivos (as poucas exceções, como o TCC e uma rodada rápida anterior, estão identificadas no texto); a seção 7 mostra o que foi reconferido a partir das {n['n_exec']} predições. Erros em **nós**; decimais com vírgula.

**Conteúdo:** 1. Resumo · 2. A execução · 3. Metodologia · 4. Resultados · 5. O que se sustenta e o que não · 6. Limitações · 7. Verificação independente · 8. Como reproduzir
"""
    doc = "\n".join([cabeca, secao_resumo(d, n, b, bens, c), secao_execucao(d, n, nb), secao_metodologia(d, n, c), secao_resultados(d, n, c, nb),
                     secao_sustenta(d, b, bens, c), secao_limites(d, n, c), secao_verificacao(ver), secao_reproduzir()])
    a.md.write_text(doc, encoding="utf-8")
    print(f"documento gravado em {a.md} ({len(doc.splitlines())} linhas) | figuras em {a.figuras}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
