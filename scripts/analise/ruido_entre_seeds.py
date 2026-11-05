#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ruido_entre_seeds.py — quanto o resultado de UMA rodada muda só por trocar a seed do treino
===========================================================================================

Contexto: o notebook rápido (`pipeline_tc_kaggle_rapido.ipynb`) treina uma seed só. Ao comparar duas rodadas dele (por exemplo,
antes e depois de mudar o modelo), é preciso saber que parte da diferença é só ruído de treino. A execução completa (5 folds ×
5 seeds × 4 variantes) mede isso: para cada variante, a métrica média dos 5 folds é calculada seed a seed, e o desvio-padrão
entre as 5 seeds é o ruído que uma rodada de 1 seed carrega.

O QUE CALCULA (por variante e métrica; fonte: `tabelas/metricas_por_execucao.csv` da execução completa)
-------------------------------------------------------------------------------------------------------
 * ``dp_media_dos_folds``   desvio-padrão, entre as 5 seeds, da métrica média dos 5 folds (o número de uma rodada de 1 seed);
 * ``dp_seeds_num_fold``    desvio-padrão entre as seeds dentro de um fold, médio sobre os folds (ruído de um fold isolado);
 * ``dp_entre_folds``       desvio-padrão entre os 5 folds da média das seeds (quanto a dificuldade do fold pesa);
 * ``tempo de uma seed``    soma das durações dos treinos de uma seed (todos os folds e variantes): o custo de uma rodada do notebook rápido;
 * ``dif_minima_2dp``       regra prática: a diferença entre DUAS rodadas independentes de 1 seed tem desvio √2 × dp_media_dos_folds,
                            então só diferenças acima de 2 × √2 × dp_media_dos_folds (≈ 95%) destoam do ruído. Não é um teste
                            estatístico: com 5 seeds o próprio desvio estimado é incerto (±35%).

MARGEM POR NÚMERO DE FOLDS (conferência da regra usada por `analisar_rodada_rapida.py`)
---------------------------------------------------------------------------------------
A regra prática vale para a média de 5 folds de 1 seed: ``RUIDO_BASE`` = 0,2 nó (RMSE de Vmax) e 0,3 nó (RMSE de ΔVmax), arredondados de
``dif_minima_2dp``. Uma rodada do notebook rápido que executa só n folds tem a média mais ruidosa, e a margem passa a ser
``RUIDO_BASE · √(5 / n)``. Para conferir essa escala, comparam-se as seeds da execução completa entre si (todos os pares de seeds, todas as
variantes e todos os subconjuntos de n folds): como são seeds do MESMO código, as diferenças são só ruído, e a fração que passa da margem é a
taxa de «falso alarme» da regra (``margem_por_folds.csv``).
Desde 2026-10-23 o notebook rápido não tem folds: uma rodada dele é um teste único (por padrão os ciclones de 2017), que corresponde ao caso
n = 1 desta tabela (≈ 0,45 nó em Vmax e ≈ 0,67 nó em ΔVmax entre duas rodadas de 1 seed no mesmo teste).

USO
---
    python3 -I scripts/analise/ruido_entre_seeds.py                       # lê resultados_pipeline/saida_pipeline, grava resultados_ruido_seeds/
    python3 -I scripts/analise/ruido_entre_seeds.py --entrada OUTRA/saida_pipeline --saida OUTRA_PASTA

SAÍDAS (em ``resultados_ruido_seeds/``)
---------------------------------------
    ruido_entre_seeds.csv    uma linha por (variante, métrica)
    ruido_entre_seeds.md     a mesma tabela em texto, com as métricas principais e a margem por número de folds
    margem_por_folds.csv     uma linha por (métrica, nº de folds): margem, mediana/p95/máximo da diferença entre seeds e fração fora da margem
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

METRICAS = {"rmse_vmax": "RMSE de Vmax (nós)", "mae_vmax": "MAE de Vmax (nós)", "rmse_dvmax": "RMSE de ΔVmax (nós)",
            "mae_dvmax": "MAE de ΔVmax (nós)", "auc_ri": "AUC de RI", "ap_ri": "AP de RI"}
RUIDO_BASE = {"rmse_vmax": 0.2, "rmse_dvmax": 0.3}      # nós, para a média de 5 folds de 1 seed (a mesma constante de `analisar_rodada_rapida.py`)
FOLDS_DA_REGRA = 5


def ruido(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for v, g in df.groupby("variante"):
        for m, nome in METRICAS.items():
            if m not in g or g[m].isna().all():
                continue
            por_seed = g.groupby("seed")[m].mean()                         # média dos folds, uma por seed
            media_folds = g.groupby("fold")[m].mean()                       # média das seeds, uma por fold
            dp_media = float(por_seed.std(ddof=1))
            linhas.append(dict(variante=v, metrica=m, descricao=nome, seeds=int(por_seed.size), folds=int(media_folds.size),
                               media=float(g[m].mean()), dp_media_dos_folds=dp_media,
                               dp_seeds_num_fold=float(g.groupby("fold")[m].std(ddof=1).mean()),
                               dp_entre_folds=float(media_folds.std(ddof=1)),
                               dif_minima_2dp=float(2 * np.sqrt(2) * dp_media)))
    return pd.DataFrame(linhas)


def margem_por_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Por métrica e nº de folds n: a margem ``RUIDO_BASE · √(5 / n)`` e a distribuição de |diferença| entre duas seeds da execução completa na média
    de n folds (todos os pares de seeds × variantes × subconjuntos de n folds). ``fora`` = quantas diferenças passam da margem."""
    seeds, folds = sorted(df["seed"].unique()), sorted(df["fold"].unique())
    linhas = []
    for m, base in RUIDO_BASE.items():
        d = df.dropna(subset=[m]).pivot_table(index=["variante", "seed"], columns="fold", values=m)
        for n in range(1, len(folds) + 1):
            difs = np.array([abs(d.loc[(v, s1), list(fs)].mean() - d.loc[(v, s2), list(fs)].mean())
                             for v in d.index.get_level_values(0).unique() for s1, s2 in itertools.combinations(seeds, 2)
                             for fs in itertools.combinations(folds, n)])
            margem = base * np.sqrt(FOLDS_DA_REGRA / n)
            linhas.append(dict(metrica=m, folds=n, comparacoes=len(difs), margem=float(margem), dif_mediana=float(np.median(difs)),
                               dif_p95=float(np.quantile(difs, 0.95)), dif_max=float(difs.max()), fora=int((difs > margem).sum()),
                               fracao_fora=float((difs > margem).mean())))
    return pd.DataFrame(linhas)


def main() -> None:
    raiz = Path(__file__).resolve().parents[2]   # raiz do projeto (este arquivo fica em scripts/analise/)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--entrada", default=str(raiz / "resultados_pipeline" / "saida_pipeline"))
    ap.add_argument("--saida", default=str(raiz / "resultados_ruido_seeds"))
    a = ap.parse_args()

    df = pd.read_csv(Path(a.entrada) / "tabelas" / "metricas_por_execucao.csv")
    tab = ruido(df)
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    tab.to_csv(saida / "ruido_entre_seeds.csv", index=False)

    ver = tab[tab["metrica"].isin(["rmse_vmax", "rmse_dvmax"])].round(3)
    linhas = ["# Ruído entre seeds (execução completa)", "",
              f"Fonte: `{a.entrada}/tabelas/metricas_por_execucao.csv` ({len(df)} execuções).", "",
              "| variante | métrica | média | dp entre seeds da média dos folds | dp entre seeds num fold | dp entre folds | diferença mínima (2·√2·dp) |",
              "|---|---|---|---|---|---|---|"]
    for r in ver.itertuples():
        linhas.append(f"| {r.variante} | {r.descricao} | {r.media:.3f} | {r.dp_media_dos_folds:.3f} | {r.dp_seeds_num_fold:.3f} | "
                      f"{r.dp_entre_folds:.3f} | {r.dif_minima_2dp:.2f} |")
    marg = margem_por_folds(df)
    marg.to_csv(saida / "margem_por_folds.csv", index=False)
    linhas += ["", f"Margem de ruído de uma seed por número de folds (regra: {RUIDO_BASE['rmse_vmax']} nó em Vmax e {RUIDO_BASE['rmse_dvmax']} nó em ΔVmax para a média de "
               f"{FOLDS_DA_REGRA} folds, escalada por √({FOLDS_DA_REGRA}/n)). Conferência com as seeds desta execução completa entre si: diferença (em valor absoluto) entre duas seeds "
               "na média de n folds, em todos os pares de seeds × variantes × subconjuntos de n folds.", "",
               "| métrica | folds | comparações | margem | diferença mediana | diferença p95 | diferença máxima | fora da margem |", "|---|---|---|---|---|---|---|---|"]
    for r in marg.itertuples():
        linhas.append(f"| {METRICAS[r.metrica]} | {r.folds} | {r.comparacoes} | {r.margem:.3f} | {r.dif_mediana:.3f} | {r.dif_p95:.3f} | {r.dif_max:.3f} | "
                      f"{r.fora} ({100 * r.fracao_fora:.1f}%) |")
    por_seed = df.groupby("seed")["segundos"].sum()
    h, m = divmod(int(round(por_seed.mean() / 60)), 60)
    linhas += ["", f"Tempo de treino de UMA seed (soma das durações dos {int((df['seed'] == df['seed'].iloc[0]).sum())} treinos: todos os folds e variantes): "
               f"{h} h {m:02d} min em média entre as {por_seed.size} seeds (mín. {por_seed.min() / 3600:.2f} h, máx. {por_seed.max() / 3600:.2f} h)."]
    (saida / "ruido_entre_seeds.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))
    print(f"\nGravado em {saida}/")


if __name__ == "__main__":
    main()
