#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_equivalencia_treino.py — confere que o `treinar_remoto.py` treina exatamente o que o notebook treina.

O script de treino remoto contém CÓPIAS das definições do `pipeline_tc_kaggle.ipynb` (configuração, janelas, tensor, modelo,
KAN, folds, treino, assinatura das execuções). Se alguém mudar uma delas só num dos dois lados, os resultados deixam de ser
comparáveis e as execuções gravadas pelo script deixam de ser reconhecidas pelo notebook (a assinatura muda). Este verificador
compara, definição por definição, a ÁRVORE SINTÁTICA (ignora comentários e espaços, mas pega qualquer mudança de código) e
termina com código 1 se achar uma diferença que não esteja declarada abaixo.

Uso (não executa nada do notebook nem do script: só lê os dois como texto/JSON):
    python3 -I scripts/verificacao/verificar_equivalencia_treino.py [--notebook pipeline_tc_kaggle.ipynb] [--script scripts/treino/treinar_remoto.py]

Rode depois de qualquer alteração em um dos dois arquivos. Se uma diferença nova for intencional, mude os dois lados ou
declare-a em DIFERENCAS_DECLARADAS (com o motivo).
"""
import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Diferenças intencionais entre o notebook e o script (nome → motivo). Qualquer outra diferença é erro.
DIFERENCAS_DECLARADAS = {
    "log": "o script também grava o log em <saida>/log_treino.txt (sobrevive à queda do SSH)",
    "salvar_execucao": "o script grava de forma atômica (arquivo temporário + os.replace; .pt antes do .npz); mesmo conteúdo",
}


def definicoes(codigo: str) -> Dict[str, ast.AST]:
    """Nome → nó, para funções, classes e atribuições simples de nível de módulo (a 1ª ocorrência de cada nome)."""
    out: Dict[str, ast.AST] = {}
    for no in ast.parse(codigo).body:
        if isinstance(no, (ast.FunctionDef, ast.ClassDef)):
            out.setdefault(no.name, no)
        elif isinstance(no, ast.Assign) and len(no.targets) == 1 and isinstance(no.targets[0], ast.Name):
            out.setdefault(no.targets[0].id, no)
    return out


def codigo_do_notebook(caminho: Path) -> Tuple[Dict[str, ast.AST], Optional[List[ast.stmt]]]:
    """Definições de todas as células de código + o corpo do `if cfg.modo == "teste":` que ajusta o Config."""
    nb = json.loads(caminho.read_text(encoding="utf-8"))
    defs: Dict[str, ast.AST] = {}
    bloco_teste = None
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        fonte = "".join(c["source"])
        try:
            arvore = ast.parse(fonte)
        except SyntaxError:
            continue
        for nome, no in definicoes(fonte).items():
            defs.setdefault(nome, no)
        if "class Config" in fonte:
            for no in arvore.body:
                if isinstance(no, ast.If) and ast.unparse(no.test) == "cfg.modo == 'teste'":
                    bloco_teste = no.body
    return defs, bloco_teste


def sem_docstring(corpo: List[ast.stmt]) -> List[ast.stmt]:
    if corpo and isinstance(corpo[0], ast.Expr) and isinstance(getattr(corpo[0], "value", None), ast.Constant) \
            and isinstance(corpo[0].value.value, str):
        return corpo[1:]
    return corpo


def main() -> int:
    raiz = Path(__file__).resolve().parents[2]   # raiz do projeto (este arquivo fica em scripts/verificacao/)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--notebook", default=str(raiz / "pipeline_tc_kaggle.ipynb"))
    p.add_argument("--script", default=str(raiz / "scripts" / "treino" / "treinar_remoto.py"))
    args = p.parse_args()

    defs_nb, bloco_teste_nb = codigo_do_notebook(Path(args.notebook))
    defs_sc = definicoes(Path(args.script).read_text(encoding="utf-8"))

    iguais, declaradas, erros, so_script = [], [], [], []
    for nome, no_sc in defs_sc.items():
        if nome not in defs_nb:
            so_script.append(nome)
        elif ast.dump(no_sc) == ast.dump(defs_nb[nome]):
            iguais.append(nome)
        elif nome in DIFERENCAS_DECLARADAS:
            declaradas.append(nome)
        else:
            erros.append(nome)

    # o bloco do modo teste (no notebook é código solto; no script é o corpo de aplicar_modo_teste)
    teste_ok = False
    if bloco_teste_nb is not None and "aplicar_modo_teste" in defs_sc:
        corpo_sc = sem_docstring(defs_sc["aplicar_modo_teste"].body)
        teste_ok = [ast.dump(n) for n in corpo_sc] == [ast.dump(n) for n in bloco_teste_nb]
    if not teste_ok:
        erros.append("aplicar_modo_teste (bloco do modo teste do Config)")

    print(f"notebook: {args.notebook}\nscript:   {args.script}\n")
    print(f"definições idênticas ({len(iguais)}): " + ", ".join(sorted(iguais)))
    print(f"\ndiferenças DECLARADAS ({len(declaradas)}):")
    for n in sorted(declaradas):
        print(f"  - {n}: {DIFERENCAS_DECLARADAS[n]}")
    print(f"\nbloco do modo teste do Config igual ao do notebook: {'sim' if teste_ok else 'NÃO'}")
    print(f"\nsó no script, sem equivalente no notebook ({len(so_script)}): " + ", ".join(sorted(so_script)))
    if erros:
        print(f"\nERRO: {len(erros)} definição(ões) diferente(s) sem declaração: " + ", ".join(erros))
        print("Mude os dois lados para ficarem iguais (ou, se for intencional, declare em DIFERENCAS_DECLARADAS).")
        return 1
    print("\nOK: o script treina exatamente o que o notebook treina.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
