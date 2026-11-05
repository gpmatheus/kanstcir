#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notebook_local.py — executa as células de código de um notebook do projeto como um script, sem Jupyter
=====================================================================================================

Para quê. Testar o notebook rápido (`pipeline_tc_kaggle_rapido.ipynb`) no próprio computador, e deixar `ajustar_hiperparametros.py`
reaproveitar o código do notebook (dados, janelas, modelo, treino) em vez de copiá-lo. As células rodam em ordem, num único espaço de
nomes, como no Kaggle; `display` cai para `print` e o Matplotlib usa o *backend* sem tela. O notebook é lido só como JSON; é código do
próprio projeto (não use com notebooks baixados).

Marcador de parada. Uma célula cujo texto começa com `# [parar-antes-desta-celula]` NÃO é executada, e a execução termina ali. No
notebook rápido ela marca o ponto entre as DEFINIÇÕES (config, dados, modelo, funções de treino) e o início do treino do protocolo.

USO (como script)
    TCIR_DIR=data/20percent TCIR_SAIDA=<pasta> TCIR_MODO=teste python3 -I scripts/treino/notebook_local.py pipeline_tc_kaggle_rapido.ipynb
        --ate-marcador       para na célula marcada (só as definições), em vez de rodar o notebook inteiro
USO (como biblioteca)
    from notebook_local import executar
    ns = executar("pipeline_tc_kaggle_rapido.ipynb", ate_marcador=True)     # devolve o espaço de nomes (ns["cfg"], ns["dados"], ...)
    ns["cfg"].epocas_max = 1                                                 # (opcional) muda algo antes de rodar o resto
    executar("pipeline_tc_kaggle_rapido.ipynb", ns=ns, a_partir_do_marcador=True)    # executa da célula-marcador até o fim

Variáveis de ambiente que o notebook lê: TCIR_DIR (pasta dos TCIR-*.h5), TCIR_SAIDA (pasta de saída), TCIR_MODO ("teste"|"completo"),
TCIR_DIVISAO ("ano"|"ciclone"), TCIR_VARIANTES (nomes separados por vírgula), TCIR_SEEDS (inteiros separados por vírgula)
e TCIR_DISPOSITIVO ("cpu"|"mps"|"cuda"; só para testes locais).
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

MARCADOR = "# [parar-antes-desta-celula]"


def celulas_de_codigo(caminho: Path) -> list:
    """[(índice da célula no notebook, texto)] só das células de código."""
    nb = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


def executar(caminho, ate_marcador: bool = False, ns: dict = None, silencioso: bool = False, a_partir_do_marcador: bool = False) -> dict:
    """Executa as células de código em ordem e devolve o espaço de nomes. Com `ate_marcador=True` para antes da célula que começa com
    `MARCADOR`; com `a_partir_do_marcador=True` pula tudo até essa célula e executa dela até o fim (use com o `ns` devolvido por uma
    chamada anterior com `ate_marcador=True`, depois de mudar o que quiser em `ns["cfg"]`: é como se dá, p. ex., um treino de 1 época
    ao notebook inteiro). Um erro numa célula é relançado com o número dela (o mesmo número do notebook, contando as células de texto)."""
    ns = {"__name__": "__main__"} if ns is None else ns
    t_ini = time.time()
    rodando = not a_partir_do_marcador
    for i, src in celulas_de_codigo(Path(caminho)):
        eh_marcador = src.lstrip().startswith(MARCADOR)
        if ate_marcador and eh_marcador:                                # sem `ate_marcador` a célula-marcador roda como qualquer outra
            if not silencioso:
                print(f"[notebook_local] parou antes da célula {i} (marcador); {time.time() - t_ini:.0f} s", flush=True)
            break
        if not rodando:
            if not eh_marcador:
                continue
            rodando = True
        t0 = time.time()
        try:
            exec(compile(src, f"<célula {i} de {Path(caminho).name}>", "exec"), ns)
        except SystemExit:
            raise
        except BaseException:
            print(f"\n[notebook_local] ERRO na célula {i}:", file=sys.stderr)
            traceback.print_exc()
            raise
        if not silencioso:
            print(f"[notebook_local] célula {i} ok ({time.time() - t0:.1f} s)", flush=True)
    if not rodando:
        raise RuntimeError(f"a_partir_do_marcador: nenhuma célula começa com «{MARCADOR}» em {caminho}")
    return ns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("notebook")
    ap.add_argument("--ate-marcador", action="store_true", help=f"para antes da célula que começa com «{MARCADOR}»")
    a = ap.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    executar(a.notebook, ate_marcador=a.ate_marcador)


if __name__ == "__main__":
    main()
