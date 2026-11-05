#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_duplicatas.py — Verificação de ciclones/imagens duplicados no TCIR
=============================================================================

Este script é INDEPENDENTE da pipeline (notebook). Ele só lê os arquivos
``TCIR-*.h5``, procura duplicatas, imprime/grava um relatório e gera um CSV que
a pipeline pode (opcionalmente) consumir para remover os registros redundantes.

POR QUE VERIFICAR DUPLICATAS?
-----------------------------
É preciso checar se o dataset tem "fotos duplicadas do mesmo
ciclone no mesmo instante de tempo" (erro de captura de satélite). Uma duplicata
é um dado redundante: a rede vê o *mesmo* exemplo mais de uma vez como se fosse
um exemplo novo. Isso tem dois efeitos ruins:

  1. "Vicia" o treino: o gradiente passa a dar peso extra a esse exemplo.
  2. Gera VAZAMENTO (data leakage) na validação cruzada: se as duas cópias
     caírem em folds diferentes, a cópia que está no treino entrega a resposta
     da cópia que está no teste, e o erro de teste fica artificialmente baixo.

O QUE O SCRIPT VERIFICA (em ordem de custo)
-------------------------------------------
  A) DUPLICATA ESTRITA (definição literal): o par (ID do ciclone, instante)
     aparece mais de uma vez. Só usa a tabela de metadados ``info`` — instantâneo.

  B) MESMO CICLONE FÍSICO COM IDs DIFERENTES: dois registros em IDs distintos,
     no MESMO instante e (praticamente) na MESMA posição (lon/lat). Acontece
     quando um ciclone cruza de uma bacia para outra (ex.: Atlântico → Pacífico
     Leste) e cada centro de previsão dá um ID. É a mesma tempestade gravada
     duas vezes. Também só usa metadados.

  C) COMPARAÇÃO DE PIXELS dos candidatos de (A) e (B): as imagens são idênticas
     bit a bit? (lê só as poucas linhas candidatas do arquivo .h5).

  D) (opcional, ``--varrer-imagens``) varredura de TODAS as imagens com hash:
     acha imagens exatamente iguais mesmo que os metadados não denunciem.
     Lê ~48 GB; leva alguns minutos. Imagens totalmente vazias (todos os pixels
     NaN) são iguais entre si, mas isso é DADO AUSENTE, não duplicata de captura:
     o relatório as separa das duplicatas e conta quantas existem.

SAÍDAS (pasta ``--saida``, padrão ``resultados_duplicatas/``)
-------------------------------------------------------------
  relatorio_duplicatas.md   relatório legível
  resumo_duplicatas.json    números principais (para conferência automática)
  duplicatas_tcir.csv       um registro por linha duplicada, com a decisão
                            (manter/remover) — é o arquivo que a pipeline lê.
                            Só é criado se houver duplicatas.

USO
---
  python scripts/dados/verificar_duplicatas.py --dados data/                     # rápido (metadados + pixels dos candidatos)
  python scripts/dados/verificar_duplicatas.py --dados data/ --varrer-imagens    # + hash de todas as imagens
  python scripts/dados/verificar_duplicatas.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # leitura de pastas somente-leitura (ex.: /kaggle/input)
import h5py
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constantes do dataset / do TCC
# --------------------------------------------------------------------------- #
# Split por ano usado no TCC (e no CNN-TC original): ciclones de 2003-2014 para
# treino, 2015-2016 para validação, 2017 para teste. O ano vem dos 4 primeiros
# caracteres do ID do ciclone (ex.: "201616L" -> 2016).
ANO_FIM_TREINO = 2014
ANO_FIM_VALIDACAO = 2016

# Ordem dos canais na matriz de imagens do TCIR: (IR, WV, VIS, PMW).
CANAL_IR, CANAL_WV, CANAL_VIS, CANAL_PMW = 0, 1, 2, 3
NOMES_CANAIS = ("IR", "WV", "VIS", "PMW")

# Recorte central 64x64 de 201x201 usado na pipeline (linhas/colunas 68..131).
RECORTE_PIPELINE = slice(68, 132)


# --------------------------------------------------------------------------- #
# Leitura da tabela de metadados (``info``) sem depender do PyTables
# --------------------------------------------------------------------------- #
class _UnpicklerRestrito(pickle.Unpickler):
    """Desserializa SOMENTE o necessário para recuperar um array numpy de objetos.

    O pandas (formato "fixed" do HDF5) guarda as colunas de texto como um pickle
    dentro do arquivo. Desserializar pickle arbitrário é inseguro; como sabemos
    exatamente o que esse pickle precisa (numpy.dtype / numpy.ndarray /
    _reconstruct), liberamos apenas esses três objetos e recusamos o resto.
    """

    _PERMITIDOS = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module, name):
        if (module, name) not in self._PERMITIDOS:
            raise pickle.UnpicklingError(f"objeto não permitido no pickle: {module}.{name}")
        if module == "numpy.core.multiarray":  # numpy 2.x renomeou o módulo para numpy._core
            try:
                from numpy._core.multiarray import _reconstruct  # type: ignore

                return _reconstruct
            except Exception:  # numpy 1.x: o caminho antigo continua valendo
                pass
        return super().find_class(module, name)


def ler_info_tcir(caminho: Path) -> pd.DataFrame:
    """Lê a tabela ``info`` de um arquivo TCIR (.h5) e devolve um DataFrame.

    Colunas: data_set (bacia), ID, lon, lat, time (AAAAMMDDHH, texto), Vmax (nós),
    R35_4qAVG, MSLP (hPa). A leitura é feita direto com h5py porque
    ``pd.read_hdf`` falha com pandas 3.x + PyTables (atributo ``pandas_type`` vem
    como bytes). O layout do pandas "fixed" é: ``axis0`` = nomes das colunas,
    ``axis1`` = índice, e blocos ``blockK_items`` / ``blockK_values`` agrupando
    colunas por tipo (float64 num bloco, objetos/strings em outro, em pickle).
    """
    with h5py.File(caminho, "r") as f:
        g = f["info"]
        ordem_colunas = [c.decode() for c in g["axis0"][:]]
        indice = g["axis1"][:]
        dados: dict[str, np.ndarray] = {}
        k = 0
        while f"block{k}_items" in g:
            nomes = [c.decode() for c in g[f"block{k}_items"][:]]
            valores = g[f"block{k}_values"][:]
            if valores.dtype == object:  # bloco de strings: vem como pickle em bytes
                valores = _UnpicklerRestrito(io.BytesIO(valores[0].tobytes())).load()
            for j, nome in enumerate(nomes):
                dados[nome] = valores[:, j]
            k += 1
    return pd.DataFrame(dados, index=indice)[ordem_colunas]


def carregar_metadados(pasta: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Lê o ``info`` de todos os TCIR-*.h5 da pasta e concatena numa tabela única."""
    arquivos = sorted(pasta.glob("TCIR-*.h5"))
    if not arquivos:
        sys.exit(f"[ERRO] nenhum arquivo TCIR-*.h5 encontrado em {pasta}")
    partes = []
    for arq in arquivos:
        df = ler_info_tcir(arq)
        df["arquivo"] = arq.name
        df["linha"] = np.arange(len(df))  # posição da linha dentro do arquivo (índice da imagem)
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    df["ano"] = df["ID"].str[:4].astype(int)
    df["split"] = np.select(
        [df["ano"] <= ANO_FIM_TREINO, df["ano"] <= ANO_FIM_VALIDACAO], ["treino", "validacao"], "teste"
    )
    df["instante"] = pd.to_datetime(df["time"], format="%Y%m%d%H")
    # CPAC usa longitude 0..360; normaliza para -180..180 antes de comparar posições.
    df["lon180"] = ((df["lon"] + 180.0) % 360.0) - 180.0
    return df, arquivos


# --------------------------------------------------------------------------- #
# Verificação A — duplicata estrita: mesmo ID + mesmo instante
# --------------------------------------------------------------------------- #
def verificar_id_instante(df: pd.DataFrame) -> pd.DataFrame:
    """Linhas cujo par (ID, instante) aparece mais de uma vez (definição literal)."""
    mascara = df.duplicated(["ID", "instante"], keep=False)
    return df[mascara].sort_values(["ID", "instante", "arquivo", "linha"])


# --------------------------------------------------------------------------- #
# Verificação B — IDs diferentes, mesmo instante, mesma posição
# --------------------------------------------------------------------------- #
def _distancia_graus(lon1, lat1, lon2, lat2) -> float:
    """Distância angular (graus) entre dois pontos da esfera — fórmula de haversine."""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(np.rad2deg(2 * np.arcsin(np.sqrt(a))))


# Pares de IDs diferentes no mesmo instante mais próximos que isto entram na tabela de "vizinhos"
# (só para o relatório mostrar que a tolerância separa bem duplicatas de ciclones vizinhos de verdade).
JANELA_CONTEXTO_GRAUS = 3.0


def verificar_ids_diferentes(df: pd.DataFrame, tol_graus: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pares de linhas com IDs diferentes, mesmo instante e posição a ≤ ``tol_graus``.

    Em cada instante só existem poucos ciclones ativos ao mesmo tempo no planeta
    (no máximo uma dezena), então comparar todos os pares dentro de cada
    instante é barato. Devolve (candidatos, vizinhos):

    - candidatos: pares com distância ≤ ``tol_graus`` (colunas i, j, dist_graus);
    - vizinhos: pares com ``tol_graus`` < distância ≤ 3° — ciclones DIFERENTES que
      estavam apenas próximos; servem para mostrar a folga da tolerância.
    """
    pares = []
    vizinhos = []
    multi = df.groupby("instante").filter(lambda g: g["ID"].nunique() > 1)
    for instante, g in multi.groupby("instante"):
        idx = g.index.to_numpy()
        ids = g["ID"].to_numpy()
        lon = g["lon180"].to_numpy()
        lat = g["lat"].to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if ids[i] == ids[j]:
                    continue
                d = _distancia_graus(lon[i], lat[i], lon[j], lat[j])
                if d <= tol_graus:
                    pares.append((idx[i], idx[j], d))
                elif d <= JANELA_CONTEXTO_GRAUS:
                    vizinhos.append((idx[i], idx[j], d))
    colunas = ["i", "j", "dist_graus"]
    return pd.DataFrame(pares, columns=colunas), pd.DataFrame(vizinhos, columns=colunas)


# --------------------------------------------------------------------------- #
# Verificação C — comparação de pixels dos candidatos
# --------------------------------------------------------------------------- #
def comparar_pixels(df: pd.DataFrame, pares: list[tuple[int, int]], pasta: Path) -> pd.DataFrame:
    """Lê as duas imagens de cada par candidato e mede o quanto se parecem.

    - ``identica``: os 4 canais são iguais bit a bit (NaN contam como iguais);
    - ``corr_ir``: correlação de Pearson do canal IR (1.0 = mesma imagem);
    - ``dif_max_ir``: maior diferença absoluta no canal IR.
    """
    abertos: dict[str, h5py.File] = {}
    linhas = []
    try:
        for i, j in pares:
            ri, rj = df.loc[i], df.loc[j]
            for r in (ri, rj):
                if r["arquivo"] not in abertos:
                    abertos[r["arquivo"]] = h5py.File(pasta / r["arquivo"], "r")
            a = abertos[ri["arquivo"]]["matrix"][int(ri["linha"])]
            b = abertos[rj["arquivo"]]["matrix"][int(rj["linha"])]
            identica = bool(np.array_equal(a, b, equal_nan=True))
            ir_a, ir_b = a[..., CANAL_IR].astype(np.float64), b[..., CANAL_IR].astype(np.float64)
            valido = np.isfinite(ir_a) & np.isfinite(ir_b)
            if valido.sum() > 1 and ir_a[valido].std() > 0 and ir_b[valido].std() > 0:
                corr = float(np.corrcoef(ir_a[valido], ir_b[valido])[0, 1])
            else:
                corr = float("nan")
            dif = float(np.nanmax(np.abs(ir_a - ir_b))) if valido.any() else float("nan")
            linhas.append({"i": i, "j": j, "identica": identica, "corr_ir": corr, "dif_max_ir": dif})
    finally:
        for f in abertos.values():
            f.close()
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# Agrupamento: quais IDs são o mesmo ciclone físico? quais linhas manter?
# --------------------------------------------------------------------------- #
class _UniaoBusca:
    """Union-Find mínimo: junta IDs que pertencem ao mesmo ciclone físico."""

    def __init__(self):
        self.pai: dict[str, str] = {}

    def achar(self, x: str) -> str:
        self.pai.setdefault(x, x)
        while self.pai[x] != x:
            self.pai[x] = self.pai[self.pai[x]]  # compressão de caminho
            x = self.pai[x]
        return x

    def unir(self, a: str, b: str) -> None:
        ra, rb = self.achar(a), self.achar(b)
        if ra != rb:
            self.pai[rb] = ra


def decidir_manter_remover(df: pd.DataFrame, pares_ids: pd.DataFrame) -> pd.DataFrame:
    """Monta a tabela final de registros duplicados com a decisão manter/remover.

    1. Une (union-find) os IDs que compartilham ao menos um par de IDs-diferentes
       candidato -> "componentes" = ciclones físicos. IDs sem par ficam sozinhos.
    2. ``id_canonico`` do componente = o ID com mais registros no dataset (empate:
       o menor ID em ordem alfabética). É o ID sob o qual a pipeline passa a tratar
       o ciclone (a série temporal é a união das linhas dos IDs equivalentes).
    3. Dentro de cada componente, linhas no MESMO instante são duplicatas. Fica
       uma por grupo: a do ID com mais registros (desempate: menor ID, arquivo, linha).
    """
    uf = _UniaoBusca()
    for _, p in pares_ids.iterrows():
        uf.unir(df.at[p["i"], "ID"], df.at[p["j"], "ID"])

    n_por_id = df.groupby("ID").size()
    raiz = df["ID"].map(uf.achar)  # raiz provisória do componente de cada linha
    # ID canônico: escolhe, em cada componente, o ID com mais linhas (empate: menor ID)
    canonico: dict[str, str] = {}
    for r, ids in df.assign(_raiz=raiz).groupby("_raiz")["ID"].unique().items():
        canonico[r] = sorted(ids, key=lambda x: (-n_por_id[x], x))[0]
    df = df.assign(id_canonico=raiz.map(canonico))

    # grupos de duplicatas: (id_canonico, instante) com mais de uma linha
    mascara = df.duplicated(["id_canonico", "instante"], keep=False)
    dup = df[mascara].copy()
    if dup.empty:
        return dup
    dup["_n_id"] = dup["ID"].map(n_por_id)
    dup = dup.sort_values(["id_canonico", "instante", "_n_id", "ID", "arquivo", "linha"],
                          ascending=[True, True, False, True, True, True])
    dup["rank"] = dup.groupby(["id_canonico", "instante"]).cumcount()
    dup["acao"] = np.where(dup["rank"] == 0, "manter", "remover")
    dup["tipo"] = np.where(
        dup.groupby(["id_canonico", "instante"])["ID"].transform("nunique") > 1,
        "IDs_diferentes_mesmo_ciclone",
        "mesmo_ID_mesmo_instante",
    )
    dup["grupo_dup"] = dup.groupby(["id_canonico", "instante"]).ngroup()
    return dup.drop(columns=["_n_id", "rank"])


# --------------------------------------------------------------------------- #
# Verificação D (opcional) — hash de todas as imagens
# --------------------------------------------------------------------------- #
def varrer_imagens(df: pd.DataFrame, pasta: Path, arquivos: list[Path], tamanho_lote: int = 128) -> pd.DataFrame:
    """Calcula dois hashes por imagem (e marca imagens sem dado de IR).

    - hash_completo: 201x201x4 inteira (qualquer duplicata exata);
    - hash_pipeline: só IR+PMW no recorte central 64x64, que é o que a rede usa;
    - ir_vazio: o canal IR do recorte é 100% NaN (satélite sem dado naquele instante).

    Imagens com hash repetido são exatamente iguais. ATENÇÃO: imagens VAZIAS (tudo NaN)
    são todas iguais entre si, mas isso é DADO AUSENTE, não duplicata de captura — por
    isso o relatório as separa. Percorre os ~48 GB do dataset em lotes sequenciais
    (leitura contígua = rápida).
    """
    h_comp = np.empty(len(df), dtype=object)
    h_pipe = np.empty(len(df), dtype=object)
    ir_vazio = np.zeros(len(df), dtype=bool)  # canal IR 100% NaN no recorte da pipeline (sem dado de satélite)
    offset = 0
    t0 = time.time()
    total_bytes = 0
    for arq in arquivos:
        with h5py.File(arq, "r") as f:
            m = f["matrix"]
            n = m.shape[0]
            for ini in range(0, n, tamanho_lote):
                lote = m[ini : ini + tamanho_lote]
                total_bytes += lote.nbytes
                for k in range(lote.shape[0]):
                    img = lote[k]
                    h_comp[offset + ini + k] = hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest()
                    recorte = np.ascontiguousarray(
                        img[RECORTE_PIPELINE, RECORTE_PIPELINE][..., [CANAL_IR, CANAL_PMW]]
                    )
                    h_pipe[offset + ini + k] = hashlib.blake2b(recorte.tobytes(), digest_size=16).hexdigest()
                    ir_vazio[offset + ini + k] = bool(np.isnan(recorte[..., 0]).all())
                if (ini // tamanho_lote) % 40 == 0:
                    dt = time.time() - t0
                    print(f"    {arq.name}: {ini + lote.shape[0]:>6}/{n}  ({total_bytes / 1e9:6.1f} GB, "
                          f"{total_bytes / 1e6 / max(dt, 1e-9):6.0f} MB/s)", flush=True)
            offset += n
    out = df[["arquivo", "linha", "ID", "time", "split"]].copy()
    out["hash_completo"] = h_comp
    out["hash_pipeline"] = h_pipe
    out["ir_vazio"] = ir_vazio
    return out


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #
def _tabela_md(df: pd.DataFrame, colunas: list[str]) -> str:
    cab = "| " + " | ".join(colunas) + " |\n|" + "|".join(["---"] * len(colunas)) + "|\n"
    corpo = "".join("| " + " | ".join(str(r[c]) for c in colunas) + " |\n" for _, r in df.iterrows())
    return (cab + corpo).rstrip("\n")


def montar_relatorio(ctx: dict) -> str:
    """Monta o relatório em markdown (blocos separados por linha em branco)."""
    df = ctx["df"]
    b: list[str] = []  # blocos de texto; são unidos com linha em branco entre eles
    b.append("# Relatório de verificação de duplicatas — TCIR")
    b.append(f"Gerado por `verificar_duplicatas.py` em {time.strftime('%Y-%m-%d %H:%M')}.")

    b.append("## 1. Dataset analisado")
    b.append(_tabela_md(ctx["resumo_arquivos"], ["arquivo", "linhas", "ciclones", "anos"]))
    b.append(f"**Total:** {len(df)} imagens e {df['ID'].nunique()} IDs de ciclone. "
             "Splits do TCC (pelo ano do ID): treino ≤ 2014, validação 2015–2016, teste 2017.")
    b.append(_tabela_md(ctx["resumo_splits"], ["split", "linhas", "ciclones"]))

    b.append("## 2. Verificação A — mesmo ID de ciclone no mesmo instante (definição literal)")
    nA = ctx["n_dup_estrita"]
    if nA == 0:
        b.append(f"**Resultado: 0 duplicatas.** Nenhum par (ID, instante) se repete nos {len(df)} registros "
                 "(nem dentro de um arquivo, nem entre arquivos).")
    else:
        b.append(f"**Resultado: {nA} linhas envolvidas em (ID, instante) repetido.**")
        b.append(_tabela_md(ctx["dup_estrita"].head(30), ["ID", "time", "arquivo", "linha"]))

    b.append("## 3. Verificação B — mesmo ciclone físico registrado com IDs diferentes")
    b.append(f"Critério: IDs diferentes, mesmo instante e posição a no máximo {ctx['tol_graus']}° "
             f"(≈ {ctx['tol_graus'] * 111:.0f} km). {ctx['texto_tolerancia']}")
    comp = ctx["componentes"]
    if comp.empty:
        b.append("**Resultado: nenhum ciclone aparece sob dois IDs.**")
    else:
        b.append(f"**Resultado: {len(comp)} ciclone(s) físico(s) gravado(s) sob mais de um ID; "
                 f"{ctx['n_remover']} registros redundantes.**")
        b.append(_tabela_md(comp, ["ids", "bacias", "linhas_por_id", "instantes_duplicados",
                                   "primeiro_instante", "ultimo_instante", "split", "imagens_identicas"]))
    b.append("Registros redundantes (a remover) por split:")
    b.append(_tabela_md(ctx["remover_por_split"], ["split", "registros_a_remover"]))

    b.append("## 4. Verificação C — comparação de pixels dos candidatos")
    if ctx["pixels"] is None or ctx["pixels"].empty:
        b.append("Sem candidatos (ou comparação de pixels desligada com `--sem-pixels`).")
    else:
        px = ctx["pixels"]
        b.append(f"Pares comparados: {len(px)}. Imagens **idênticas bit a bit nos 4 canais: "
                 f"{int(px['identica'].sum())}** de {len(px)}.")
        nao = px[~px["identica"]]
        if len(nao):
            b.append("Pares NÃO idênticos (mesmo instante e posição quase igual, mas o recorte está levemente "
                     "deslocado — `corr_ir` é a correlação de Pearson do canal IR; `dif_max_ir`, a maior "
                     "diferença em Kelvin). São *quase-duplicatas* e também são tratadas como redundantes:")
            b.append(_tabela_md(nao.round(4), ["i", "j", "corr_ir", "dif_max_ir"]))

    b.append("## 5. Verificação D — varredura de hash de todas as imagens")
    if ctx["hash"] is None:
        b.append("Não executada (use `--varrer-imagens`).")
    else:
        h = ctx["hash"]
        b.append(f"Imagens lidas: {h['n_imagens']}. Cada imagem foi resumida em um hash (impressão digital): "
                 "hashes iguais = imagens exatamente iguais, independentemente do que os metadados dizem.")
        b.append(f"- Grupos de imagens **completas** (201×201×4) idênticas: **{h['grupos_completo']}** "
                 f"({h['linhas_em_grupos_completo']} linhas), dos quais:\n"
                 f"  - {h['completo_por_duplicata_ID']} são as duplicatas de ID das verificações A/B (pares do mesmo ciclone físico);\n"
                 f"  - {h['completo_por_imagem_vazia']} são **imagens vazias** (tudo NaN) repetidas "
                 f"({h['completo_linhas_imagem_vazia']} linhas) — dado ausente, não duplicata de captura;\n"
                 f"  - **{h['grupos_completo_nao_explicados']} não explicados**.\n"
                 f"- Grupos de recortes **IR+PMW 64×64** idênticos (o que a rede enxerga): **{h['grupos_pipeline']}** "
                 f"({h['linhas_em_grupos_pipeline']} linhas): {h['pipeline_por_duplicata_ID']} duplicatas de ID, "
                 f"{h['pipeline_por_imagem_vazia']} de imagem vazia e **{h['grupos_pipeline_nao_explicados']} não explicados**.")
        b.append(f"**Dado ausente (não é duplicata, mas afeta a pipeline):** {h['imagens_com_ir_vazio']} imagens "
                 f"({100 * h['imagens_com_ir_vazio'] / h['n_imagens']:.2f}% do dataset, em {h['ciclones_com_ir_vazio']} "
                 f"ciclones) têm o canal IR 100% ausente no recorte central — por split: "
                 f"treino {h['ir_vazio_por_split']['treino']}, validação {h['ir_vazio_por_split']['validacao']}, "
                 f"teste {h['ir_vazio_por_split']['teste']}. Com a limpeza do TCC (NaN → 0) elas viram imagens "
                 "totalmente zeradas que ainda carregam um rótulo de Vmax; a pipeline as trata como **frames "
                 "inválidos** e descarta as janelas que as contêm.")
        if h["detalhe_nao_explicados"]:
            b.append(h["detalhe_nao_explicados"])

    b.append("## 6. Veredito")
    b.append(ctx["veredito"])
    b.append("## 7. Como a pipeline usa este resultado")
    b.append("`duplicatas_tcir.csv` lista cada registro duplicado com `acao` = manter/remover e o "
             "`id_canonico` do ciclone físico. O notebook (i) remove as linhas marcadas `remover`, "
             "(ii) passa a tratar todos os IDs de um mesmo `id_canonico` como UM ciclone — a série temporal é "
             "a união das linhas — e (iii) usa o `id_canonico` como grupo na validação cruzada, para que as duas "
             "cópias nunca caiam em folds diferentes (sem vazamento).")
    return "\n\n".join(b) + "\n"


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Verifica duplicatas de ciclones/imagens no TCIR.")
    ap.add_argument("--dados", type=Path, default=Path("data"), help="pasta com os TCIR-*.h5 (padrão: ./data)")
    ap.add_argument("--saida", type=Path, default=Path("resultados_duplicatas"), help="pasta de saída")
    ap.add_argument("--tol-graus", type=float, default=0.25,
                    help="distância máx. (graus) para considerar a mesma posição (padrão 0,25° ≈ 28 km)")
    ap.add_argument("--varrer-imagens", action="store_true",
                    help="calcula hash de TODAS as imagens (lê ~48 GB; alguns minutos)")
    ap.add_argument("--sem-pixels", action="store_true", help="não abre as imagens (só metadados)")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/5] Lendo metadados de {args.dados} ...")
    df, arquivos = carregar_metadados(args.dados)
    resumo_arquivos = (df.groupby("arquivo")
                       .agg(linhas=("ID", "size"), ciclones=("ID", "nunique"),
                            ano_min=("ano", "min"), ano_max=("ano", "max"))
                       .reset_index())
    resumo_arquivos["anos"] = resumo_arquivos["ano_min"].astype(str) + "–" + resumo_arquivos["ano_max"].astype(str)
    resumo_splits = (df.groupby("split").agg(linhas=("ID", "size"), ciclones=("ID", "nunique"))
                     .reindex(["treino", "validacao", "teste"]).reset_index())
    print(f"      {len(df)} imagens | {df['ID'].nunique()} IDs | arquivos: {[a.name for a in arquivos]}")

    print("[2/5] Verificação A — mesmo ID + mesmo instante ...")
    dup_estrita = verificar_id_instante(df)
    print(f"      linhas envolvidas: {len(dup_estrita)}")

    print(f"[3/5] Verificação B — IDs diferentes, mesmo instante, posição ≤ {args.tol_graus}° ...")
    pares_ids, vizinhos = verificar_ids_diferentes(df, args.tol_graus)
    print(f"      pares candidatos: {len(pares_ids)} (maior distância: "
          f"{pares_ids['dist_graus'].max() if len(pares_ids) else float('nan'):.3f}°) | "
          f"vizinhos mais próximos fora da tolerância: "
          f"{vizinhos['dist_graus'].min() if len(vizinhos) else float('nan'):.2f}°")

    pixels = None
    if len(pares_ids) and not args.sem_pixels:
        print("[4/5] Verificação C — comparando pixels dos candidatos ...")
        pixels = comparar_pixels(df, list(zip(pares_ids["i"], pares_ids["j"])), args.dados)
        print(f"      idênticos bit a bit: {int(pixels['identica'].sum())}/{len(pixels)}")
    else:
        print("[4/5] Verificação C — pulada (sem candidatos ou --sem-pixels).")

    dup = decidir_manter_remover(df, pares_ids)
    n_remover = int((dup["acao"] == "remover").sum()) if len(dup) else 0

    # tabela-resumo por componente (ciclone físico com mais de um ID)
    componentes = pd.DataFrame()
    if len(dup):
        multi = dup[dup["tipo"] == "IDs_diferentes_mesmo_ciclone"]
        linhas = []
        for canon, g in multi.groupby("id_canonico"):
            ids_comp = sorted(g["ID"].unique())  # todo ID do componente tem ao menos um registro duplicado
            sub = df[df["ID"].isin(ids_comp)]
            n_inst = g["instante"].nunique()
            ident = "—"
            if pixels is not None and len(pixels):
                idx_g = set(g.index)
                px = pixels[pixels["i"].isin(idx_g) & pixels["j"].isin(idx_g)]
                ident = f"{int(px['identica'].sum())}/{len(px)}"
            linhas.append({
                "ids": " + ".join(ids_comp),
                "bacias": " + ".join(sub.groupby("ID")["data_set"].first().loc[ids_comp]),
                "linhas_por_id": " / ".join(str(int((df["ID"] == i).sum())) for i in ids_comp),
                "instantes_duplicados": n_inst,
                "primeiro_instante": g["instante"].min().strftime("%Y-%m-%d %Hh"),
                "ultimo_instante": g["instante"].max().strftime("%Y-%m-%d %Hh"),
                "split": "/".join(sorted(sub["split"].unique())),
                "imagens_identicas": ident,
            })
        componentes = pd.DataFrame(linhas)
    remover_por_split = (dup[dup["acao"] == "remover"].groupby("split").size()
                         .reindex(["treino", "validacao", "teste"], fill_value=0)
                         .rename("registros_a_remover").reset_index()
                         if len(dup) else
                         pd.DataFrame({"split": ["treino", "validacao", "teste"], "registros_a_remover": [0, 0, 0]}))

    resultado_hash = None
    if args.varrer_imagens:
        print("[5/5] Verificação D — varrendo hash de todas as imagens (isso lê todo o dataset) ...")
        hs = varrer_imagens(df, args.dados, arquivos)
        explicadas = set(map(tuple, dup[["arquivo", "linha"]].to_numpy())) if len(dup) else set()
        vazias = set(map(tuple, hs.loc[hs["ir_vazio"], ["arquivo", "linha"]].to_numpy()))

        def _grupos_repetidos(col: str) -> pd.DataFrame:
            """Linhas cujo hash aparece em mais de uma imagem."""
            return hs[hs.duplicated(col, keep=False)]

        def _classificar(grupos: pd.DataFrame, col: str) -> dict:
            """Separa os grupos de hash em: duplicata de ID (A/B), imagem vazia e não explicados."""
            cont = {"duplicata_ID": 0, "imagem_vazia": 0, "linhas_vazias": 0, "nao_explicados": []}
            for _, g in grupos.groupby(col):
                chaves = list(zip(g["arquivo"], g["linha"]))
                if all(c in explicadas for c in chaves):
                    cont["duplicata_ID"] += 1
                elif all(c in vazias for c in chaves):
                    cont["imagem_vazia"] += 1
                    cont["linhas_vazias"] += len(g)
                else:
                    cont["nao_explicados"].append(g)
            return cont

        g_comp, g_pipe = _grupos_repetidos("hash_completo"), _grupos_repetidos("hash_pipeline")
        c_comp, c_pipe = _classificar(g_comp, "hash_completo"), _classificar(g_pipe, "hash_pipeline")
        ne = c_comp["nao_explicados"] + c_pipe["nao_explicados"]
        detalhe = ""
        if ne:
            amostra = pd.concat(ne).drop_duplicates(["arquivo", "linha"])
            detalhe = ("Amostra das linhas em grupos não explicados:\n\n"
                       + _tabela_md(amostra[["arquivo", "linha", "ID", "time", "split"]].head(40),
                                    ["arquivo", "linha", "ID", "time", "split"]))
        vz = hs[hs["ir_vazio"]]
        vazias_por_split = (vz.groupby("split").size().reindex(["treino", "validacao", "teste"], fill_value=0))
        resultado_hash = {
            "n_imagens": int(len(hs)),
            "grupos_completo": int(g_comp["hash_completo"].nunique()),
            "linhas_em_grupos_completo": int(len(g_comp)),
            "completo_por_duplicata_ID": c_comp["duplicata_ID"],
            "completo_por_imagem_vazia": c_comp["imagem_vazia"],
            "completo_linhas_imagem_vazia": c_comp["linhas_vazias"],
            "grupos_completo_nao_explicados": len(c_comp["nao_explicados"]),
            "grupos_pipeline": int(g_pipe["hash_pipeline"].nunique()),
            "linhas_em_grupos_pipeline": int(len(g_pipe)),
            "pipeline_por_duplicata_ID": c_pipe["duplicata_ID"],
            "pipeline_por_imagem_vazia": c_pipe["imagem_vazia"],
            "grupos_pipeline_nao_explicados": len(c_pipe["nao_explicados"]),
            "imagens_com_ir_vazio": int(len(vz)),
            "ciclones_com_ir_vazio": int(vz["ID"].nunique()),
            "ir_vazio_por_split": {k: int(v) for k, v in vazias_por_split.items()},
            "detalhe_nao_explicados": detalhe,
        }
        print(f"      grupos de imagens idênticas (completas): {resultado_hash['grupos_completo']} = "
              f"{c_comp['duplicata_ID']} duplicatas de ID + {c_comp['imagem_vazia']} grupo(s) de imagem vazia "
              f"({c_comp['linhas_vazias']} linhas) + {len(c_comp['nao_explicados'])} não explicados")
        print(f"      imagens com IR 100% ausente: {len(vz)} ({100 * len(vz) / len(hs):.2f}% do dataset), "
              f"em {vz['ID'].nunique()} ciclones")
    else:
        print("[5/5] Verificação D — pulada (use --varrer-imagens).")

    # ---------------- texto que justifica a tolerância (derivado dos próprios dados) ----------------
    if len(pares_ids):
        texto_tol = (f"Nos dados, os {len(pares_ids)} pares candidatos ficam a no máximo "
                     f"{pares_ids['dist_graus'].max():.3f}° um do outro")
    else:
        texto_tol = "Nenhum par ficou dentro dessa tolerância"
    if len(vizinhos):
        texto_tol += (f"; já o par de ciclones DIFERENTES mais próximo fora da tolerância está a "
                      f"{vizinhos['dist_graus'].min():.2f}° — há folga entre 'duplicata' e 'vizinho'.")
    else:
        texto_tol += "."

    # ---------------- veredito em linguagem direta ----------------
    n_estrita = len(dup_estrita)
    partes = []
    partes.append(
        "- **Definição literal (mesmo ID + mesmo instante): "
        + ("NÃO há duplicatas (0 linhas)." if n_estrita == 0 else f"HÁ duplicatas ({n_estrita} linhas).")
        + "**"
    )
    if n_remover:
        ids_afetados = ", ".join(componentes["ids"]) if len(componentes) else "—"
        iguais = (f"{int(pixels['identica'].sum())} de {len(pixels)} pares têm imagens idênticas bit a bit; "
                  "os demais são quase-duplicatas (recorte deslocado)") if pixels is not None and len(pixels) else \
            "pixels não comparados"
        partes.append(
            f"- **Há {n_remover} registros redundantes** em {len(componentes)} ciclone(s) físico(s) que aparecem "
            f"gravados sob dois IDs ({ids_afetados}): mesmo instante e mesma posição; {iguais}. Devem ser "
            "removidos do treino, e os dois IDs devem ficar juntos no mesmo fold da validação cruzada "
            "(senão uma cópia vai para o treino e a outra para o teste = vazamento)."
        )
    else:
        partes.append("- Nenhum ciclone físico aparece sob mais de um ID.")
    if resultado_hash is not None:
        nao = resultado_hash["grupos_completo_nao_explicados"] + resultado_hash["grupos_pipeline_nao_explicados"]
        partes.append(
            "- Varredura por hash de todas as imagens: "
            + ("todas as imagens repetidas são explicadas pelas duplicatas acima ou por imagens vazias; "
               "não há nenhuma outra duplicata exata." if nao == 0 else
               f"{nao} grupo(s) de imagens idênticas NÃO explicados — veja a seção 5.")
        )
        partes.append(
            f"- **Dado ausente:** {resultado_hash['imagens_com_ir_vazio']} imagens "
            f"({100 * resultado_hash['imagens_com_ir_vazio'] / resultado_hash['n_imagens']:.2f}%) têm o IR 100% NaN. "
            "Não são duplicatas, mas a pipeline precisa descartá-las (frames inválidos)."
        )
    veredito = "\n".join(partes)

    ctx = dict(df=df, resumo_arquivos=resumo_arquivos, resumo_splits=resumo_splits,
               n_dup_estrita=n_estrita, dup_estrita=dup_estrita, tol_graus=args.tol_graus,
               texto_tolerancia=texto_tol,
               componentes=componentes, n_remover=n_remover, remover_por_split=remover_por_split,
               pixels=pixels, hash=resultado_hash, veredito=veredito)
    relatorio = montar_relatorio(ctx)
    (args.saida / "relatorio_duplicatas.md").write_text(relatorio, encoding="utf-8")

    resumo = {
        "total_imagens": int(len(df)),
        "total_ids": int(df["ID"].nunique()),
        "duplicatas_estritas_id_instante_linhas": int(n_estrita),
        "pares_ids_diferentes_mesmo_ciclone": int(len(pares_ids)),
        "ciclones_fisicos_com_dois_ids": int(len(componentes)),
        "registros_a_remover": n_remover,
        "registros_a_remover_por_split": {r["split"]: int(r["registros_a_remover"])
                                          for _, r in remover_por_split.iterrows()},
        "pares_com_pixels_identicos": int(pixels["identica"].sum()) if pixels is not None and len(pixels) else None,
        "varredura_hash": ({k: v for k, v in resultado_hash.items() if k != "detalhe_nao_explicados"}
                           if resultado_hash else None),
        "tolerancia_graus": args.tol_graus,
        "segundos": round(time.time() - t0, 1),
    }
    (args.saida / "resumo_duplicatas.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(dup):
        cols = ["grupo_dup", "tipo", "acao", "id_canonico", "ID", "time", "data_set", "lon", "lat", "Vmax",
                "split", "arquivo", "linha"]
        dup.sort_values(["grupo_dup", "acao"], ascending=[True, True])[cols].to_csv(
            args.saida / "duplicatas_tcir.csv", index=False)
    else:
        # sem duplicatas: remove um CSV antigo para a pipeline não usar resultado obsoleto
        (args.saida / "duplicatas_tcir.csv").unlink(missing_ok=True)

    print("\n" + "=" * 78)
    print("VEREDITO")
    print("=" * 78)
    print(veredito)
    print("=" * 78)
    print(f"Relatório : {args.saida / 'relatorio_duplicatas.md'}")
    print(f"Resumo    : {args.saida / 'resumo_duplicatas.json'}")
    if len(dup):
        print(f"CSV       : {args.saida / 'duplicatas_tcir.csv'}  (consumido pela pipeline)")
    print(f"Tempo total: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
