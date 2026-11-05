#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_assinatura_execucoes.py — os notebooks atuais reconhecem (pelo nome) as execuções de uma pasta `runs/`?
=================================================================================================================

Cada execução é gravada como `runs/<variante>__f<fold>__s<seed>__<assinatura>.npz`. A assinatura é um hash curto dos hiperparâmetros
que mudam o resultado (todos os campos de `Config`, menos os de `_CAMPOS_SO_DE_EXECUCAO`) e do conjunto de janelas. Um notebook só
reaproveita execuções cuja assinatura é igual à que ele calcula. A lista `_CAMPOS_SO_DE_EXECUCAO` mudou depois da 2ª execução do Kaggle
(cinco campos operacionais, como o orçamento de tempo, saíram da assinatura); então a mesma configuração tem assinaturas diferentes
na regra que rodou lá e na regra dos notebooks de hoje.

O QUE FAZ (só lê arquivos; o notebook baixado é lido como JSON/AST, nunca executado)
 1. lê o `config.json` da pasta de saída e reconstrói as janelas (mesmas regras do notebook; `estatisticas_janelas.py`);
 2. lê `_CAMPOS_SO_DE_EXECUCAO` do notebook que rodou (o `__notebook__.ipynb` baixado) e do notebook atual (`pipeline_tc_kaggle.ipynb`);
 3. recalcula a assinatura com cada regra e compara com o sufixo dos `runs/*.npz`.

USO
    python3 -I scripts/verificacao/verificar_assinatura_execucoes.py [--entrada resultados_pipeline/saida_pipeline]
                                                 [--notebook-executado resultados_pipeline/__notebook__.ipynb]
                                                 [--notebook-atual pipeline_tc_kaggle.ipynb] [--dados data]
SAÍDA: `resultados_assinatura/assinatura.md` (e o mesmo texto no terminal).

Se a assinatura gravada não for a do notebook atual, as execuções continuam válidas (é o mesmo experimento), mas não são reconhecidas pelo
nome: para reaproveitá-las, copie os arquivos trocando o sufixo `__<assinatura antiga>` pelo da regra atual (conteúdo idêntico).
"""
import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]  # raiz do projeto (este arquivo fica em scripts/verificacao/)
sys.path.insert(0, str(RAIZ / "scripts" / "dados"))  # com `python3 -I` a pasta do script não entra sozinha em sys.path
import estatisticas_janelas as ej           # noqa: E402  (mesmas regras de janelas do notebook)


def campos_so_de_execucao(caminho: Path) -> set:
    """Lê o literal `_CAMPOS_SO_DE_EXECUCAO = {...}` de um notebook (como JSON + AST; nada é executado)."""
    nb = json.loads(caminho.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        for no in ast.parse("".join(c["source"])).body:
            if isinstance(no, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_CAMPOS_SO_DE_EXECUCAO" for t in no.targets):
                return {e.value for e in no.value.elts}
    raise SystemExit(f"_CAMPOS_SO_DE_EXECUCAO não encontrado em {caminho}")


def assinatura(cfg: dict, excluidos: set, hash_janelas: str) -> str:
    d = {k: v for k, v in cfg.items() if k not in excluidos}
    d["janelas"] = hash_janelas
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--entrada", default=str(RAIZ / "resultados_pipeline" / "saida_pipeline"))
    ap.add_argument("--notebook-executado", default=str(RAIZ / "resultados_pipeline" / "__notebook__.ipynb"))
    ap.add_argument("--notebook-atual", default=str(RAIZ / "pipeline_tc_kaggle.ipynb"))
    ap.add_argument("--dados", default=str(RAIZ / "data"))
    ap.add_argument("--cache", default=str(RAIZ / "resultados_janelas" / "frac_invalido_ir.csv"))
    ap.add_argument("--saida", default=str(RAIZ / "resultados_assinatura"))
    a = ap.parse_args()

    entrada = Path(a.entrada)
    cfg = json.loads((entrada / "config.json").read_text(encoding="utf-8"))
    sufixos = sorted({m[1] for p in (entrada / "runs").glob("*.npz") if (m := re.search(r"__([0-9a-f]{8})\.npz$", p.name))})

    arqs = ej.localizar_arquivos(Path(a.dados))
    tabela = ej.carregar_tabela(arqs)
    frac = ej.obter_frac_invalido_ir(tabela, arqs, Path(a.cache), refazer=False)
    jan, _ = ej.construir_janelas(tabela, frac)
    hash_janelas = hashlib.md5("|".join(jan["id_ciclone"] + "@" + jan["instante"].astype(str)).encode()).hexdigest()

    regra_executada = campos_so_de_execucao(Path(a.notebook_executado))
    regra_atual = campos_so_de_execucao(Path(a.notebook_atual))
    sig_exec, sig_atual = assinatura(cfg, regra_executada, hash_janelas), assinatura(cfg, regra_atual, hash_janelas)

    linhas = ["# Assinatura das execuções gravadas × regra dos notebooks atuais", "",
              f"- pasta: `{a.entrada}` ({len(list((entrada / 'runs').glob('*.npz')))} execuções); sufixos encontrados: {', '.join(f'`{s}`' for s in sufixos)}",
              f"- janelas reconstruídas: {len(jan)} (hash `{hash_janelas[:12]}…`)",
              f"- regra do notebook que rodou (`{Path(a.notebook_executado).name}`): `{sig_exec}` — "
              + ("**confere** com o sufixo gravado" if sig_exec in sufixos else "NÃO confere com o sufixo gravado"),
              f"- regra do notebook atual (`{Path(a.notebook_atual).name}`): `{sig_atual}`",
              "- campos que saíram da assinatura depois da execução: " + (", ".join(f"`{c}`" for c in sorted(regra_atual - regra_executada)) or "nenhum"), ""]
    if sig_atual in sufixos:
        linhas.append("Os notebooks atuais reconhecem estas execuções pelo nome.")
    else:
        linhas.append(f"Os notebooks atuais **não** reconhecem estas execuções pelo nome (esperam o sufixo `__{sig_atual}`). O conteúdo é o mesmo experimento: "
                      f"para reaproveitá-las, copie os arquivos trocando `__{sufixos[0] if sufixos else '<antigo>'}` por `__{sig_atual}`.")
    texto = "\n".join(linhas) + "\n"
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "assinatura.md").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Gravado em {saida}/assinatura.md")


if __name__ == "__main__":
    main()
