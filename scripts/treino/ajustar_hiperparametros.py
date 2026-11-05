#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ajustar_hiperparametros.py — ajuste de hiperparâmetros da proposta (cnn_lstm_kan) com o código do notebook rápido
==================================================================================================================

O QUE FAZ. Carrega as DEFINIÇÕES do notebook rápido (config, dados, janelas, modelo e função de treino, até a célula marcada com
`# [parar-antes-desta-celula]`, via `notebook_local.py`) e treina a proposta com várias combinações de hiperparâmetros de TREINO e
REGULARIZAÇÃO, na divisão por ano do TCC (treino 2003–2014, validação 2015–2016). Cada candidato é medido SÓ na validação: o conjunto de
teste (2017) nunca é previsto aqui (`avaliar_teste=False`), então a escolha não pode vazar o teste. Não usa folds: com a validação fixa
(2015–2016: ≈ 9 mil janelas na base inteira) basta um treino por candidato.

O QUE NÃO MUDA. A arquitetura (CNN encoder → LSTM → cabeças KAN, 2 tarefas), as entradas e o tamanho das KAN ficam como estão: só entram
hiperparâmetros de treino e de regularização (taxa de aprendizado, *weight decay*, dropout, probabilidade de rotação, peso de ΔVmax).

PLANOS (`--plano`)
    velocidade   um treino de 1 época com a configuração padrão: mede segundos por época (dimensiona a busca)
    cruz         varredura em cruz (um hiperparâmetro de cada vez, em volta do padrão) + combinação dos vencedores + confirmação:
                   etapa 0  padrão com 2 seeds (mede o ruído de seed/GPU: a margem abaixo da qual uma diferença não vale)
                   etapa 1  cada valor alternativo, seed 0  (ESPACO, abaixo)
                   etapa 2  combinação dos que melhoram a RMSE de Vmax de validação (qualquer melhora) sem piorar ΔVmax além de
                            TOLERANCIA_DVMAX, com 2 seeds; passa se melhorar pelo menos metade da margem de ruído (se a combinação não
                            passa, segue o melhor vencedor isolado, desde que ele sozinho melhore pelo menos metade da margem)
                   etapa 3  confirmação com o número de épocas cheio (`--epocas-finais`), só do que passou: padrão × candidato (seed 0);
                            só então o padrão é trocado (de novo exigindo metade da margem e ΔVmax dentro da tolerância); se a
                            combinação não se confirma, confirma também o melhor vencedor isolado
    O critério é a RMSE de Vmax na validação, na melhor época (a tarefa principal; é o mesmo critério do *early stopping*).

USO
    python3 -I scripts/treino/ajustar_hiperparametros.py --dados <pasta com os TCIR-*.h5> --dispositivo mps --epocas 12 --epocas-finais 25
    (é retomável: o `ajuste.csv` de `--saida` guarda cada execução ao terminar; rodar de novo pula o que já foi feito)

SAÍDAS (em `--saida`, padrão `resultados_ajuste/<nome>/`)
    ajuste.csv              uma linha por treino: etapa, candidato, mudanças, seed, épocas, melhor época, RMSE de validação (Vmax e ΔVmax),
                            RMSE de treino em modo de avaliação, folga, segundos
    RELATORIO_AJUSTE.md     ranking, efeito de cada hiperparâmetro contra o padrão, margem de ruído e a configuração escolhida
    melhor_config.json      a configuração escolhida (só os campos que diferem do padrão do notebook)
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]  # raiz do projeto (este arquivo fica em scripts/treino/)
sys.path.insert(0, str(RAIZ / "scripts" / "treino"))  # com `python3 -I` a pasta do script não entra sozinha em sys.path
import notebook_local                      # noqa: E402

VARIANTE = "cnn_lstm_kan"
# valores alternativos de cada hiperparâmetro (o padrão é o do notebook, lido de `cfg`); `peso_dvmax` = pesos_tarefas["dvmax"]
ESPACO = {                                  # um valor alternativo por hiperparâmetro (o custo local é de ≈ 14 min por treino de 10 épocas)
    "dropout": [0.3],                       # padrão 0.1: a rede decora o treino (RMSE de treino 5,99 × validação 9,56 em Vmax; 7,06 × 16,02 em ΔVmax)
    "weight_decay": [5e-2],                 # padrão 1e-4 (AdamW: o decaimento por passo é lr × wd, então só valores altos pesam)
    "prob_rotacao": [1.0],                  # padrão 0.55: mais aumento de dados (a rotação só vale no treino)
    "peso_dvmax": [0.2],                    # padrão 0.5: a multitarefa piorou o Vmax e a cabeça de ΔVmax decora o treino
    "lr": [2e-3],                           # padrão 1e-3: com a metade dos passos por época (menos dados locais) um pico maior pode compensar
}
TOLERANCIA_DVMAX = 0.30      # nós: um candidato não pode piorar a RMSE de ΔVmax na validação mais que isso contra o padrão
MARGEM_MINIMA = 0.15         # nós: piso da margem de ruído (a margem é o maior entre este piso e a diferença entre as 2 seeds do padrão)


def mudancas_para_config(cfg0, mudancas: dict):
    """Config nova = a do notebook com as mudanças aplicadas (`peso_dvmax` entra em `pesos_tarefas`)."""
    mud = dict(mudancas)
    if "peso_dvmax" in mud:
        mud["pesos_tarefas"] = {**cfg0.pesos_tarefas, "dvmax": mud.pop("peso_dvmax")}
    return dataclasses.replace(cfg0, **mud)


def linha_do_resultado(etapa: str, nome: str, mudancas: dict, seed: int, cfg_i, res: dict) -> dict:
    """Uma linha de `ajuste.csv` a partir do resultado de `treinar_execucao`."""
    h = res["meta"]["historico"]
    m = h[res["meta"]["melhor_epoca"]]
    val_dv = [x.get("rmse_val_dvmax") for x in h if x.get("rmse_val_dvmax") is not None and np.isfinite(x.get("rmse_val_dvmax"))]
    return dict(etapa=etapa, candidato=nome, mudancas=json.dumps(mudancas, sort_keys=True), seed=seed, epocas_max=cfg_i.epocas_max,
                melhor_epoca=res["meta"]["melhor_epoca"] + 1, n_epocas=res["meta"]["n_epocas"],
                rmse_val_vmax=res["meta"]["rmse_val"], rmse_val_dvmax=m.get("rmse_val_dvmax"),
                rmse_val_dvmax_min=min(val_dv) if val_dv else np.nan,
                rmse_treino_vmax=m.get("rmse_treino_vmax"), rmse_treino_dvmax=m.get("rmse_treino_dvmax"),
                folga_vmax=res["meta"]["rmse_val"] - m.get("rmse_treino_vmax"),
                folga_dvmax=(m.get("rmse_val_dvmax") - m.get("rmse_treino_dvmax")) if m.get("rmse_val_dvmax") is not None else np.nan,
                segundos=res["meta"]["segundos"])


class Ajuste:
    def __init__(self, ns: dict, saida: Path, epocas: int, epocas_finais: int):
        self.ns, self.saida = ns, saida
        self.cfg0 = ns["cfg"]
        self.epocas, self.epocas_finais = epocas, epocas_finais
        self.csv = saida / "ajuste.csv"
        self.tabela = pd.read_csv(self.csv, dtype={"etapa": str, "candidato": str, "mudancas": str}) if self.csv.exists() else pd.DataFrame()

    # -- execução de um candidato (retomável) --------------------------------------------------------------------------------------
    def rodar(self, etapa: str, nome: str, mudancas: dict, seed: int, epocas: int) -> dict:
        mud = {**mudancas, "epocas_max": epocas}
        chave = json.dumps(mud, sort_keys=True)
        if len(self.tabela):
            ja = self.tabela[(self.tabela["etapa"] == etapa) & (self.tabela["candidato"] == nome) & (self.tabela["seed"] == seed)
                             & (self.tabela["epocas_max"] == epocas)]
            if len(ja):
                return ja.iloc[-1].to_dict()
        cfg_i = mudancas_para_config(self.cfg0, mud)
        t0 = time.time()
        res = self.ns["treinar_execucao"](VARIANTE, seed, self.ns["dados"], self.ns["particao"], cfg_i, avaliar_teste=False)
        lin = linha_do_resultado(etapa, nome, {k: v for k, v in mudancas.items()}, seed, cfg_i, res)
        self.tabela = pd.concat([self.tabela, pd.DataFrame([lin])], ignore_index=True)
        self.tabela.to_csv(self.csv, index=False)
        print(f"[ajuste] {etapa:>2} {nome:<28s} seed {seed} épocas {epocas:>2} | val Vmax {lin['rmse_val_vmax']:.3f} ΔVmax {lin['rmse_val_dvmax']:.3f} | "
              f"treino Vmax {lin['rmse_treino_vmax']:.2f} | melhor época {lin['melhor_epoca']}/{lin['n_epocas']} | {(time.time() - t0) / 60:.1f} min", flush=True)
        del res
        return lin

    def media(self, etapa: str, nome: str, epocas: int, col: str = "rmse_val_vmax") -> float:
        g = self.tabela[(self.tabela["etapa"] == etapa) & (self.tabela["candidato"] == nome) & (self.tabela["epocas_max"] == epocas)]
        return float(g[col].mean()) if len(g) else float("nan")


def plano_velocidade(aj: Ajuste) -> None:
    cfg_i = mudancas_para_config(aj.cfg0, {"epocas_max": 1})
    t0 = time.time()
    res = aj.ns["treinar_execucao"](VARIANTE, 0, aj.ns["dados"], aj.ns["particao"], cfg_i, avaliar_teste=False)
    h = res["meta"]["historico"][0]
    p = aj.ns["particao"]
    print(f"[velocidade] 1 época = {h['segundos']:.1f} s ({len(p.treino)} janelas de treino, {len(p.val)} de validação; 1 treino de 1 época levou "
          f"{time.time() - t0:.1f} s no total) | val Vmax {res['meta']['rmse_val']:.2f}")
    for e in (12, 25):
        print(f"[velocidade] estimativa de 1 treino de {e} épocas: {e * h['segundos'] / 60:.1f} min")


def plano_cruz(aj: Ajuste) -> None:
    c0, E, EF = aj.cfg0, aj.epocas, aj.epocas_finais
    padrao = {"lr": c0.lr, "weight_decay": c0.weight_decay, "dropout": c0.dropout, "prob_rotacao": c0.prob_rotacao,
              "peso_dvmax": c0.pesos_tarefas["dvmax"]}
    print("[ajuste] padrão do notebook:", padrao, "| épocas de triagem:", E, "| épocas da confirmação:", EF, flush=True)

    # etapa 0: o padrão com 2 seeds → margem de ruído
    for s in (0, 1):
        aj.rodar("0", "padrao", {}, s, E)
    g0 = aj.tabela[(aj.tabela["etapa"] == "0") & (aj.tabela["epocas_max"] == E)]
    margem = max(MARGEM_MINIMA, float(g0["rmse_val_vmax"].max() - g0["rmse_val_vmax"].min()))
    ref_v, ref_d = float(g0["rmse_val_vmax"].mean()), float(g0["rmse_val_dvmax"].mean())
    print(f"[ajuste] padrão: val Vmax {ref_v:.3f} (2 seeds: {g0['rmse_val_vmax'].round(3).tolist()}), ΔVmax {ref_d:.3f}; margem de ruído = {margem:.3f} nó", flush=True)

    # etapa 1: um hiperparâmetro de cada vez (seed 0)
    fracos = {}                                                       # os que melhoram algo (> 0) sem piorar ΔVmax além da tolerância
    for hp, valores in ESPACO.items():
        for v in valores:
            lin = aj.rodar("1", f"{hp}={v:g}", {hp: v}, 0, E)
            melhora, pior_dv = ref_v - lin["rmse_val_vmax"], lin["rmse_val_dvmax"] - ref_d     # melhora > 0 = melhor que o padrão
            print(f"[ajuste]    {hp}={v:g}: Vmax {-melhora:+.3f} nó (margem {margem:.3f}) | ΔVmax {pior_dv:+.3f} nó", flush=True)
            if melhora > 0 and pior_dv <= TOLERANCIA_DVMAX and (hp not in fracos or lin["rmse_val_vmax"] < fracos[hp][1]):
                fracos[hp] = (v, lin["rmse_val_vmax"])
    print("[ajuste] melhoram algo sem piorar ΔVmax além da tolerância:", {k: v[0] for k, v in fracos.items()} or "nenhum", flush=True)

    # etapa 2: a combinação deles (2 seeds); se ela não passa, o melhor vencedor isolado (que só tem a seed 0 da etapa 1)
    unico = None                                                      # (mudanças, melhora de Vmax contra o padrão) do melhor vencedor isolado
    if fracos:
        hp_u = min(fracos, key=lambda h: fracos[h][1])
        unico = ({hp_u: fracos[hp_u][0]}, ref_v - fracos[hp_u][1])
    candidato = {hp: v[0] for hp, v in fracos.items()}
    ok_triagem = False
    if len(candidato) >= 2:
        for s in (0, 1):
            aj.rodar("2", "combinada", candidato, s, E)
        comb_v, comb_d = aj.media("2", "combinada", E), aj.media("2", "combinada", E, "rmse_val_dvmax")
        ok_triagem = ref_v - comb_v >= margem / 2 and comb_d - ref_d <= TOLERANCIA_DVMAX
        print(f"[ajuste] combinada {candidato}: val Vmax {comb_v:.3f} (padrão {ref_v:.3f}) ΔVmax {comb_d:.3f} (padrão {ref_d:.3f}) → "
              f"{'passa' if ok_triagem else 'não passa'} na triagem (melhora ≥ metade da margem)", flush=True)
        if not ok_triagem and unico[1] >= margem / 2:
            candidato, ok_triagem = unico[0], True
            print(f"[ajuste] a combinação não passa; segue o melhor vencedor isolado {candidato} (melhora de {unico[1]:+.3f} nó na triagem)", flush=True)
    elif len(candidato) == 1:
        ok_triagem = unico[1] >= margem / 2
        print(f"[ajuste] único candidato {unico[0]}: melhora de {unico[1]:+.3f} nó → {'passa' if ok_triagem else 'não passa'} na triagem", flush=True)

    # etapa 3: confirmação com as épocas cheias (padrão × candidato), só do que passou na triagem; troca o padrão só se confirmar
    confirmacoes, escolhida = [], {}

    def confirmar(mud: dict) -> bool:
        nome = "+".join(f"{hp}={v:g}" for hp, v in mud.items())
        aj.rodar("3", "padrao", {}, 0, EF)
        aj.rodar("3", nome, mud, 0, EF)
        v_pad, v_esc = aj.media("3", "padrao", EF), aj.media("3", nome, EF)
        d_pad, d_esc = aj.media("3", "padrao", EF, "rmse_val_dvmax"), aj.media("3", nome, EF, "rmse_val_dvmax")
        passou = v_pad - v_esc >= margem / 2 and d_esc - d_pad <= TOLERANCIA_DVMAX
        confirmacoes.append(dict(candidato=mud, rmse_val_vmax_padrao=v_pad, rmse_val_vmax_candidato=v_esc, rmse_val_dvmax_padrao=d_pad,
                                 rmse_val_dvmax_candidato=d_esc, passou=bool(passou)))
        print(f"[ajuste] confirmação ({EF} épocas): padrão val Vmax {v_pad:.3f} ΔVmax {d_pad:.3f} × {nome} Vmax {v_esc:.3f} ΔVmax {d_esc:.3f} → "
              f"{'confirmado' if passou else 'não confirmado'} (melhora de Vmax ≥ {margem / 2:.3f} nó e ΔVmax dentro da tolerância)", flush=True)
        return passou

    if ok_triagem:
        if confirmar(candidato):
            escolhida = candidato
        elif unico is not None and unico[0] != candidato and unico[1] >= margem / 2:
            print("[ajuste] a combinação não se confirma com as épocas cheias; confirma o melhor vencedor isolado", flush=True)
            if confirmar(unico[0]):
                escolhida = unico[0]
    else:
        print("[ajuste] nenhum candidato passou na triagem: o padrão do notebook continua (sem treino de confirmação).", flush=True)
    (aj.saida / "melhor_config.json").write_text(json.dumps(dict(escolhida=escolhida, padrao=padrao, margem_de_ruido=margem, candidato=candidato,
                                                                  confirmacoes=confirmacoes, epocas_triagem=E, epocas_confirmacao=EF),
                                                             indent=2, ensure_ascii=False), encoding="utf-8")
    escrever_relatorio(aj, padrao, escolhida, margem, confirmacoes)


def escrever_relatorio(aj: Ajuste, padrao: dict, escolhida: dict, margem: float, confirmacoes: list) -> None:
    t = aj.tabela.copy()
    E = aj.epocas
    cols = ["etapa", "candidato", "seed", "epocas_max", "melhor_epoca", "rmse_val_vmax", "rmse_val_dvmax", "rmse_treino_vmax", "rmse_treino_dvmax",
            "folga_vmax", "folga_dvmax", "segundos"]
    L = ["# Ajuste de hiperparâmetros da proposta (`cnn_lstm_kan`)", "",
         f"- **Divisão:** {aj.ns['particao'].descricao}; treino {len(aj.ns['particao'].treino)} janelas, validação {len(aj.ns['particao'].val)}. "
         "O teste não foi previsto em nenhum treino do ajuste.",
         f"- **Dispositivo:** {aj.ns['device']}; triagem com {aj.epocas} épocas, confirmação com {aj.epocas_finais}.",
         f"- **Padrão do notebook:** {json.dumps(padrao, ensure_ascii=False)}",
         f"- **Margem de ruído** (diferença entre as 2 seeds do padrão, com piso de {MARGEM_MINIMA} nó): {margem:.3f} nó de RMSE de Vmax na validação.",
         f"- **Regra:** um valor só vira o novo padrão se, no treino com as épocas cheias, a RMSE de Vmax na validação melhora pelo menos metade da margem "
         f"({margem / 2:.3f} nó) e a de ΔVmax não piora mais que {TOLERANCIA_DVMAX} nó.",
         f"- **Escolhida:** {json.dumps(escolhida, ensure_ascii=False) if escolhida else 'nenhuma mudança (o padrão continua)'}", ""]
    g0 = t[(t["etapa"] == "0") & (t["epocas_max"] == E)]
    g1 = t[(t["etapa"] == "1") & (t["epocas_max"] == E)]
    if len(g0) and len(g1):
        ref_v, ref_d = g0["rmse_val_vmax"].mean(), g0["rmse_val_dvmax"].mean()
        L += [f"## Efeito de cada valor alternativo (etapa 1, seed 0) contra o padrão (média de {len(g0)} seeds: Vmax {ref_v:.3f}, ΔVmax {ref_d:.3f})", "",
              "Diferença = candidato − padrão, em nós: **negativa = melhor que o padrão**. Folga = RMSE de validação − RMSE de treino em modo de avaliação (quanto maior, mais sobreajuste).", "",
              "| candidato | RMSE Vmax | Δ Vmax | RMSE ΔVmax | Δ ΔVmax | melhor época | folga Vmax | folga ΔVmax |", "|---|---|---|---|---|---|---|---|"]
        for _, r in g1.sort_values("rmse_val_vmax").iterrows():
            L.append(f"| {r['candidato']} | {r['rmse_val_vmax']:.3f} | {r['rmse_val_vmax'] - ref_v:+.3f} | {r['rmse_val_dvmax']:.3f} | {r['rmse_val_dvmax'] - ref_d:+.3f} | "
                     f"{int(r['melhor_epoca'])}/{int(r['n_epocas'])} | {r['folga_vmax']:.2f} | {r['folga_dvmax']:.2f} |")
        L += [f"| *padrão (média de {len(g0)} seeds)* | {ref_v:.3f} | 0 | {ref_d:.3f} | 0 | {g0['melhor_epoca'].mean():.1f}/{int(g0['n_epocas'].mean())} | "
              f"{g0['folga_vmax'].mean():.2f} | {g0['folga_dvmax'].mean():.2f} |", ""]
    if confirmacoes:
        L += ["## Confirmação com as épocas cheias (seed 0)", "", "| candidato | Vmax padrão | Vmax candidato | ΔVmax padrão | ΔVmax candidato | confirmado |", "|---|---|---|---|---|---|"]
        for c in confirmacoes:
            L.append(f"| {json.dumps(c['candidato'], ensure_ascii=False)} | {c['rmse_val_vmax_padrao']:.3f} | {c['rmse_val_vmax_candidato']:.3f} | "
                     f"{c['rmse_val_dvmax_padrao']:.3f} | {c['rmse_val_dvmax_candidato']:.3f} | {'sim' if c['passou'] else 'não'} |")
        L.append("")
    L += ["## Todos os treinos (RMSE em nós; treino em modo de avaliação; menor é melhor)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t[cols].iterrows():
        L.append("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r.tolist()) + " |")
    (aj.saida / "RELATORIO_AJUSTE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[ajuste] relatório:", aj.saida / "RELATORIO_AJUSTE.md", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--notebook", default=str(RAIZ / "pipeline_tc_kaggle_rapido.ipynb"))
    ap.add_argument("--dados", required=True, help="pasta com os TCIR-*.h5 (use uma pasta só com os 3 arquivos que quer; a busca é recursiva)")
    ap.add_argument("--saida", default=str(RAIZ / "resultados_ajuste" / "local"))
    ap.add_argument("--dispositivo", default="", help="cpu, mps ou cuda (vazio = o que o notebook escolher)")
    ap.add_argument("--modo", default="completo", choices=["completo", "teste"], help="`teste` = 36 ciclones (só para conferir o script)")
    ap.add_argument("--plano", default="cruz", choices=["velocidade", "cruz"])
    ap.add_argument("--epocas", type=int, default=12, help="épocas de cada candidato na triagem (etapas 0 a 2)")
    ap.add_argument("--epocas-finais", type=int, default=25, help="épocas da confirmação (etapa 3)")
    a = ap.parse_args()

    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    os.environ.update(TCIR_MODO=a.modo, TCIR_DIR=str(a.dados), TCIR_SAIDA=str(saida / "notebook"), TCIR_VARIANTES=VARIANTE, TCIR_SEEDS="0",
                      TCIR_DIVISAO="ano" if a.modo == "completo" else "ciclone")
    if a.dispositivo:
        os.environ["TCIR_DISPOSITIVO"] = a.dispositivo
    t0 = time.time()
    ns = notebook_local.executar(a.notebook, ate_marcador=True, silencioso=True)
    print(f"[ajuste] definições do notebook carregadas em {time.time() - t0:.0f} s | dispositivo {ns['device']} | divisão: {ns['particao'].descricao}", flush=True)
    aj = Ajuste(ns, saida, a.epocas, a.epocas_finais)
    (plano_velocidade if a.plano == "velocidade" else plano_cruz)(aj)


if __name__ == "__main__":
    main()
