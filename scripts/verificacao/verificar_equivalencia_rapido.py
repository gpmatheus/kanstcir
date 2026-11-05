#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_equivalencia_rapido.py — confere o que o notebook RÁPIDO tem de igual e de diferente em relação ao notebook COMPLETO
==============================================================================================================================

Desde 2026-10-23 o notebook rápido (`pipeline_tc_kaggle_rapido.ipynb`) NÃO usa folds: uma divisão única (treino / validação / teste, por padrão por
ano como no TCC) e um treino por (variante, seed). O notebook completo (`pipeline_tc_kaggle.ipynb`) segue com a validação cruzada em 5 folds. Logo
os dois já não são o mesmo EXPERIMENTO (as assinaturas das execuções diferem e uma não reaproveita a outra). O que este verificador garante é que
continuam sendo o mesmo MODELO e o mesmo NÚCLEO DE TREINO — e lista, uma a uma, as diferenças de protocolo, para nenhuma passar sem declaração.

Compara, definição por definição, a ÁRVORE SINTÁTICA dos dois notebooks (ignora comentários e espaços, mas pega qualquer mudança de código) e
termina com código 1 se encontrar uma diferença que não esteja declarada abaixo. O que é verificado:
  * DEFINIÇÕES IDÊNTICAS: modelo (`CodificadorCNN`, `CamadaKAN`, `CabecaKAN`, `CabecaMLP`, `ModeloTC`), tensor de entrada (`montar_lote`,
    `rotacionar_e_recortar`), janelas e duplicatas, perda e métricas, `prever`, `VARIANTES`... Tudo o que não estiver declarado como diferente tem de ser igual;
  * NÚCLEO DO TREINO (`treinar_execucao`): o laço de épocas (otimizador, agenda OneCycle, corte de gradiente, perda, early stopping) e as definições
    de normalização, modelo, otimizador e agenda são iguais ao do completo depois de tirar a instrumentação das curvas; o que difere (assinatura, a
    divisão dos dados no lugar do fold, a semente, o bloco final do teste) é só protocolo;
  * `Config`: o rápido só REMOVE os campos de folds, só ACRESCENTA os da divisão e do painel, e os demais campos são iguais — exceto os
    hiperparâmetros AJUSTADOS (`HIPERPARAMETROS_AJUSTADOS`), cujos valores diferentes aparecem na saída (o completo não foi ajustado);
  * `_CAMPOS_SO_DE_EXECUCAO`: o rápido só troca `folds_executar` pelos campos novos que não mudam o resultado;
  * o bloco do modo teste e `cfg = Config(...)`: iguais ao do completo, menos as linhas e argumentos de folds, seeds e divisão.

Use-o depois de editar qualquer um dos dois notebooks: se aparecer uma diferença de MODELO ou de NÚCLEO DE TREINO sem declaração, a mudança foi só num lado
e os dois deixaram de treinar o mesmo modelo.

Uso (só lê os dois arquivos como JSON/texto; não executa nada deles):
    python3 -I scripts/verificacao/verificar_equivalencia_rapido.py [--completo pipeline_tc_kaggle.ipynb] [--rapido pipeline_tc_kaggle_rapido.ipynb]
"""
import argparse
import ast
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Campos de `Config` que o rápido REMOVEU (folds) e que ACRESCENTOU (rótulo da rodada, diagnóstico das curvas e a divisão dos dados).
CAMPOS_REMOVIDOS_CONFIG = {"n_folds", "folds_executar", "frac_validacao_interna"}
CAMPOS_NOVOS_CONFIG = {"nome_experimento", "notas", "n_janelas_aval_treino", "divisao", "ano_fim_treino", "ano_fim_validacao", "frac_validacao", "frac_teste"}
# Novos campos que NÃO entram na assinatura (não mudam o resultado): os demais (a divisão) entram, pois mudam o experimento.
CAMPOS_NOVOS_SO_DE_EXECUCAO = {"nome_experimento", "notas", "n_janelas_aval_treino"}
# Hiperparâmetros que o ajuste de 2026-10-23 pode ter mudado SÓ no rápido (o completo mantém os valores originais). Ficam no `Config` como valores padrão.
HIPERPARAMETROS_AJUSTADOS = {"lr", "weight_decay", "dropout", "prob_rotacao", "pesos_tarefas", "epocas_max", "paciencia"}
# Argumentos de `Config(...)` que só o rápido passa (vêm do painel de controle).
ARGUMENTOS_NOVOS_CFG = {"seeds", "variantes", "variante_referencia", "nome_experimento", "notas", "log_por_epoca", "divisao"}
# Argumentos de `Config(...)` que só o completo passa (folds).
ARGUMENTOS_REMOVIDOS_CFG = {"folds_executar"}

# Diferenças declaradas, sem verificação estrutural (nome → motivo). Qualquer outra diferença é erro.
DIFERENCAS_DECLARADAS = {
    "DIR_SAIDA": "uma pasta de saída por rodada (<nome>_seed<seed>) em vez de saida_pipeline",
    "ZIP_SAIDA": "o pacote se chama resultados_<nome>_seed<seed>.zip e leva só a pasta da rodada",
    "cores": "paleta única por variante (COR_VARIANTE) em vez de cores por posição; só estética",
    "carregar_tabela_bruta": "só a mensagem do aviso sobre duplicatas (fala em «treino e teste» e não em «validação cruzada»)",
    # --- protocolo sem folds
    "executar_protocolo": "protocolo sem folds: o plano é seed × variante e o treino recebe a divisão (`particao`) no lugar de `folds`",
    "resumo_final": "o resumo registra a divisão dos dados e os tamanhos dos conjuntos no lugar de `n_folds` e `folds`; leva a chave `experimento`",
    "nome_run": "o nome do arquivo não tem mais `f<fold>`: <variante>__s<seed>__<assinatura>",
    "carregar_execucao": "a execução não tem mais a chave `fold`",
    "carregar_todas_execucoes": "só entram nas tabelas e figuras as execuções do plano da rodada (variantes e seeds do painel); sem filtro de folds",
    "n_pendentes": "a chamada do protocolo passa `particao` em vez de `folds_janela` (e fica na célula marcada para o ajuste de hiperparâmetros)",
    "tabela_execucoes": "uma linha por (variante, seed), sem `fold`",
    "resumo_por_variante": "com 1 seed (sem desvio) mostra só o valor, em vez de «± 0,00»",
    "predicoes_agrupadas": "o conjunto de teste é um só: sem `fold`",
    "_execucao_baseline": "os baselines têm uma execução só (sem `fold`)",
    "baselines_dvmax": "os baselines se ajustam no treino da divisão e se avaliam no mesmo teste (um só, sem folds)",
    "comparar_por_janela": "Wilcoxon por janela por seed (sem fold)",
    "comparar_bootstrap": "o sorteio de ciclones é só entre os ciclones do TESTE (`idx_teste`): com folds todos os ciclones eram testados, agora não",
    "carregar_modelo_treinado": "pesos da 1ª seed, sem `fold` no nome",
    "_janelas_do_teste": "sem `fold`",
    "explicar_gradcam": "sem `fold`",
    "explicar_kan": "sem `fold`",
    "explicar_latentes": "sem `fold`",
}
# Definições do completo que o rápido não tem (cada uma com o motivo).
REMOVIDAS_DECLARADAS = {
    "RAIZ": "o zip do rápido usa RAIZ_ZIP e NOME_PACOTE",
    "IGNORAR_DIRS": "o zip do rápido filtra as pastas na própria varredura",
    "dividir_folds": "folds removidos do rápido: a divisão é `dividir_treino_val_teste`",
    "folds_janela": "folds removidos do rápido",
    "tab_folds": "folds removidos do rápido (no lugar, `tab_particao`)",
    "separar_indices": "folds removidos do rápido (a validação deixou de ser sorteada dentro do treino de cada fold)",
    "comparar_por_execucao": "folds removidos: o teste «por execução» era pseudorreplicação (seeds × folds)",
    "comparar_por_fold": "folds removidos: não há «1 ponto por fold»",
    "folds_rodados": "folds removidos do rápido",
    "_fold0": "folds removidos do rápido",
    "desenhar_desempenho_por_fold": "folds removidos: no lugar, `desenhar_validacao_vs_teste`",
    "testes_exec": "folds removidos (ver `comparar_por_execucao`)",
    "testes_fold": "folds removidos (ver `comparar_por_fold`)",
}
# Atribuições e o laço de épocas de `treinar_execucao` que formam o NÚCLEO do treino (têm de ser iguais nos dois notebooks).
NUCLEO_ATRIBUICOES = {"ytr_v", "ytr_d", "norm", "yv_n", "yd_n", "modelo", "opt", "passos", "agenda", "scaler", "pesos", "idx_tr_t", "y_val", "historico"}


def definicoes(codigo: str) -> Dict[str, ast.AST]:
    """Nome → nó, para funções, classes e atribuições simples de nível de módulo (a 1ª ocorrência de cada nome)."""
    out: Dict[str, ast.AST] = {}
    for no in ast.parse(codigo).body:
        if isinstance(no, (ast.FunctionDef, ast.ClassDef)):
            out.setdefault(no.name, no)
        elif isinstance(no, ast.Assign) and len(no.targets) == 1 and isinstance(no.targets[0], ast.Name):
            out.setdefault(no.targets[0].id, no)
    return out


def ler_notebook(caminho: Path) -> Tuple[Dict[str, ast.AST], Optional[List[ast.stmt]]]:
    """Definições de todas as células de código + o corpo do `if cfg.modo == "teste":` que ajusta o Config."""
    nb = json.loads(caminho.read_text(encoding="utf-8"))
    defs: Dict[str, ast.AST] = {}
    bloco_teste = None
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        fonte = "".join(c["source"])
        arvore = ast.parse(fonte)
        for nome, no in definicoes(fonte).items():
            defs.setdefault(nome, no)
        if "class Config" in fonte:
            for no in arvore.body:
                if isinstance(no, ast.If) and ast.unparse(no.test) == "cfg.modo == 'teste'":
                    bloco_teste = no.body
    return defs, bloco_teste


class TiraInstrumentacao(ast.NodeTransformer):
    """Remove de `treinar_execucao` o que o rápido acrescentou só para registrar as curvas: não mexe no treino em si."""
    NOMES = {"n_aval", "idx_tr_aval", "extra"}

    def visit_Assign(self, no):
        if len(no.targets) == 1 and isinstance(no.targets[0], ast.Name) and no.targets[0].id in self.NOMES:
            return None
        return self.generic_visit(no)

    def visit_If(self, no):
        if ast.unparse(no.test) == "cfg.log_por_epoca and (ep + 1) % cfg.log_por_epoca == 0":      # o log por época
            return None
        return self.generic_visit(no)

    def visit_Call(self, no):
        self.generic_visit(no)
        no.keywords = [k for k in no.keywords if not (k.arg is None and ast.unparse(k.value) == "extra")]   # historico.append(dict(..., **extra))
        return no


def nucleo_do_treino(funcao: ast.FunctionDef) -> List[str]:
    """O NÚCLEO de `treinar_execucao`, depois de tirar a instrumentação: as atribuições de `NUCLEO_ATRIBUICOES` e o laço de épocas, na ordem."""
    corpo = TiraInstrumentacao().visit(copy.deepcopy(funcao)).body
    saida = []
    for s in corpo:
        alvo = s.targets[0].id if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) else None
        # (`historico, melhor, ... = [], np.inf, ...` é uma atribuição de tupla: entra pelo texto do alvo)
        if alvo in NUCLEO_ATRIBUICOES or (isinstance(s, ast.Assign) and "historico" in ast.unparse(s.targets[0])):
            saida.append(ast.dump(s))
        elif isinstance(s, ast.For) and ast.unparse(s.iter) == "range(cfg.epocas_max)":
            saida.append(ast.dump(s))
    return saida


def sem_docstring(corpo: List[ast.stmt]) -> List[ast.stmt]:
    if corpo and isinstance(corpo[0], ast.Expr) and isinstance(getattr(corpo[0], "value", None), ast.Constant) \
            and isinstance(corpo[0].value.value, str):
        return corpo[1:]
    return corpo


def campos_config(no_classe: ast.ClassDef) -> Dict[str, str]:
    return {s.target.id: ast.dump(s) for s in no_classe.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}


def valor_do_campo(no_classe: ast.ClassDef, nome: str) -> str:
    for s in no_classe.body:
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id == nome and s.value is not None:
            return ast.unparse(s.value)
    return "?"


def campos_so_de_execucao(no: ast.AST) -> set:
    return {e.value for e in no.value.elts} if isinstance(no, ast.Assign) and isinstance(no.value, ast.Set) else set()


def main() -> int:
    raiz = Path(__file__).resolve().parents[2]   # raiz do projeto (este arquivo fica em scripts/verificacao/)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--completo", default=str(raiz / "pipeline_tc_kaggle.ipynb"))
    p.add_argument("--rapido", default=str(raiz / "pipeline_tc_kaggle_rapido.ipynb"))
    args = p.parse_args()

    d_c, teste_c = ler_notebook(Path(args.completo))
    d_r, teste_r = ler_notebook(Path(args.rapido))
    erros: List[str] = []
    estruturais: List[Tuple[str, str]] = []

    def conferir(nome: str, ok: bool, motivo: str) -> None:
        (estruturais if ok else erros).append((nome, motivo) if ok else f"{nome}: {motivo}")

    # ---- Config: o rápido só remove os campos de folds, só acrescenta os novos, e os demais são iguais (menos os hiperparâmetros ajustados)
    cc, cr = campos_config(d_c["Config"]), campos_config(d_r["Config"])
    removidos, novos = set(cc) - set(cr), set(cr) - set(cc)
    alterados = sorted(n for n in cc if n in cr and cc[n] != cr[n])
    ajustados = [n for n in alterados if n in HIPERPARAMETROS_AJUSTADOS]
    inesperados = [n for n in alterados if n not in HIPERPARAMETROS_AJUSTADOS]
    problemas = ([f"campos do completo alterados sem declaração: {inesperados}"] if inesperados else []) + \
                ([f"campos removidos diferentes do declarado: {sorted(removidos)} (declarado: {sorted(CAMPOS_REMOVIDOS_CONFIG)})"] if removidos != CAMPOS_REMOVIDOS_CONFIG else []) + \
                ([f"campos novos diferentes do declarado: {sorted(novos)} (declarado: {sorted(CAMPOS_NOVOS_CONFIG)})"] if novos != CAMPOS_NOVOS_CONFIG else [])
    conferir("Config", not problemas,
             f"removidos (folds) = {sorted(removidos)}; novos = {sorted(novos)}; demais campos iguais"
             + (f"; hiperparâmetros ajustados no rápido: " + ", ".join(f"{n} {valor_do_campo(d_c['Config'], n)} → {valor_do_campo(d_r['Config'], n)}" for n in ajustados) if ajustados else "")
             if not problemas else "; ".join(problemas))
    sc, sr = campos_so_de_execucao(d_c["_CAMPOS_SO_DE_EXECUCAO"]), campos_so_de_execucao(d_r["_CAMPOS_SO_DE_EXECUCAO"])
    esperado = (sc - CAMPOS_REMOVIDOS_CONFIG) | CAMPOS_NOVOS_SO_DE_EXECUCAO
    conferir("_CAMPOS_SO_DE_EXECUCAO", sr == esperado,
             "sai `folds_executar`; entram só os campos que não mudam o resultado (rótulo e diagnóstico); a divisão dos dados ENTRA na assinatura"
             if sr == esperado else f"diferença inesperada: {sorted(sr ^ esperado)}")

    # ---- bloco do modo teste: igual, menos as linhas de folds/seeds (completo) e as da divisão (rápido)
    if teste_c is not None and teste_r is not None:
        def linha_de_folds_ou_seeds(n):
            t = ast.unparse(n)
            return t.startswith(("cfg.n_folds", "cfg.folds_executar")) or t == "cfg.seeds = (0, 1)"
        def linha_da_divisao(n):
            return ast.unparse(n).startswith(("cfg.divisao", "cfg.frac_validacao"))
        a = [ast.dump(n) for n in teste_c if not linha_de_folds_ou_seeds(n)]
        b = [ast.dump(n) for n in teste_r if not linha_da_divisao(n)]
        conferir("bloco do modo teste", a == b, "igual ao do completo, sem as linhas de folds/seeds (e com as da divisão por ciclone)")
    else:
        erros.append("bloco do modo teste: não encontrado em um dos notebooks")

    # ---- cfg = Config(...): os argumentos do completo seguem iguais, menos os de folds
    kc = {k.arg: ast.dump(k.value) for k in d_c["cfg"].value.keywords}
    kr = {k.arg: ast.dump(k.value) for k in d_r["cfg"].value.keywords}
    faltam_no_rapido = set(kc) - set(kr)
    extras = set(kr) - set(kc)
    ok_cfg = all(kr.get(k) == v for k, v in kc.items() if k not in ARGUMENTOS_REMOVIDOS_CFG) and extras <= ARGUMENTOS_NOVOS_CFG and faltam_no_rapido <= ARGUMENTOS_REMOVIDOS_CFG
    conferir("cfg = Config(...)", ok_cfg,
             f"argumentos do completo iguais, menos os de folds {sorted(faltam_no_rapido)}; extras do painel = {sorted(extras)}" if ok_cfg
             else f"argumentos inesperados: extras {sorted(extras - ARGUMENTOS_NOVOS_CFG)}, faltando {sorted(faltam_no_rapido - ARGUMENTOS_REMOVIDOS_CFG)}")

    # ---- NÚCLEO do treino
    if "treinar_execucao" in d_c and "treinar_execucao" in d_r:
        nc, nr = nucleo_do_treino(d_c["treinar_execucao"]), nucleo_do_treino(d_r["treinar_execucao"])
        conferir("treinar_execucao (núcleo)", nc == nr and len(nc) > 0,
                 f"{len(nc)} blocos iguais ao do completo (normalização dos alvos, modelo, AdamW, agenda OneCycle, perda, laço de épocas e early stopping), "
                 "depois de tirar a instrumentação das curvas; difere só o protocolo (assinatura, divisão no lugar do fold, semente, bloco final do teste)"
                 if nc == nr and nc else f"o núcleo do treino DIFERE do completo ({len(nc)} blocos no completo, {len(nr)} no rápido)")
    else:
        erros.append("treinar_execucao: falta em um dos notebooks")

    # ---- o resto, definição por definição
    ja_verificadas = {"Config", "_CAMPOS_SO_DE_EXECUCAO", "cfg", "treinar_execucao"}
    iguais, declaradas, novas, removidas = [], [], [], []
    for nome, no in d_r.items():
        if nome in ja_verificadas:
            continue
        if nome not in d_c:
            novas.append(nome)
        elif ast.dump(no) == ast.dump(d_c[nome]):
            iguais.append(nome)
        elif nome in DIFERENCAS_DECLARADAS:
            declaradas.append(nome)
        else:
            erros.append(f"{nome}: diferente do completo e não declarado")
    for nome in d_c:
        if nome not in d_r and nome not in ja_verificadas:
            if nome in REMOVIDAS_DECLARADAS:
                removidas.append(nome)
            else:
                erros.append(f"{nome}: existe no completo e sumiu do rápido (não declarado)")

    print(f"completo: {args.completo}\nrápido:   {args.rapido}\n")
    print(f"definições idênticas ({len(iguais)}): " + ", ".join(sorted(iguais)))
    print("\nverificadas por estrutura:")
    for n, m in estruturais:
        print(f"  - {n}: {m}")
    print(f"\ndiferenças de protocolo DECLARADAS ({len(declaradas)}):")
    for n in sorted(declaradas):
        print(f"  - {n}: {DIFERENCAS_DECLARADAS[n]}")
    print(f"\nsó no rápido ({len(novas)}): " + ", ".join(sorted(novas)))
    print(f"\nsó no completo, declaradas ({len(removidas)}):")
    for n in sorted(removidas):
        print(f"  - {n}: {REMOVIDAS_DECLARADAS[n]}")
    if erros:
        print(f"\nERRO: {len(erros)} diferença(s) sem declaração — o rápido e o completo NÃO treinam mais o mesmo modelo/núcleo de treino:")
        for e in erros:
            print("  -", e)
        print("Se a mudança é intencional, tudo bem, mas decida onde ela vale: só no rápido (e declare aqui) ou nos dois (mude o completo e `treinar_remoto.py`).")
        return 1
    print("\nOK: o modelo, o pré-processamento, as métricas e o núcleo do treino são os mesmos nos dois notebooks. "
          "O protocolo é diferente de propósito (rápido sem folds; completo com 5 folds): as execuções de um NÃO são reconhecidas pelo outro.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
