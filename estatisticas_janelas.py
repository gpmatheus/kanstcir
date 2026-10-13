#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estatisticas_janelas.py — Contagens e estatísticas das janelas usadas pela pipeline
===================================================================================

Este script é INDEPENDENTE do notebook (como o ``verificar_duplicatas.py``). Ele reproduz, sem PyTorch, as
regras do notebook ``pipeline_tc_kaggle.ipynb`` que decidem QUAIS janelas existem e calcula os números que a
``DOCUMENTACAO_PIPELINE.md`` cita no resumo (seções "Números reais" e 6.5):

  * nº de imagens, de ciclones e de quadros inválidos;
  * nº de janelas utilizáveis, janelas com rótulo de ΔVmax e janelas de intensificação rápida (RI);
  * **média e desvio padrão do Vmax das janelas** (e mais: mediana, quantis, categorias de intensidade);
  * divisão treino/validação/teste do TCC, janelas por ciclone e a composição dos folds da validação cruzada.

COMO O Vmax DAS JANELAS É CALCULADO
-----------------------------------
1. Cada janela é uma sequência de quadros consecutivos (a cada 3 h) do mesmo ciclone. O rótulo ``vmax`` da janela
   é o Vmax (em nós, do *best track* que acompanha o TCIR) do ÚLTIMO quadro, o instante ``t``.
2. Junta-se o rótulo de TODAS as janelas utilizáveis (uma janela = uma amostra, e não um ciclone) e calcula-se a
   média e o desvio padrão. Ciclones longos geram mais janelas e pesam mais.
3. O desvio padrão tem duas versões: amostral (ddof=1, a do ``describe()`` do pandas usado no notebook, e a
   publicada na documentação) e populacional (ddof=0, que é o RMSE de um "modelo" que sempre prevê a média). Com
   65 mil janelas a diferença é desprezível, mas as duas ficam gravadas.

POR QUE O SCRIPT LÊ AS IMAGENS (E NÃO SÓ A TABELA ``info``)?
------------------------------------------------------------
Um quadro com mais de 10% de pixels de IR ausentes é inválido e sai da série; como a janela não pode atravessar
buracos, saber QUAIS janelas existem exige a fração de IR ausente de cada imagem. Essa varredura lê ~48 GB e
demora (dezenas de minutos). Ela é feita UMA vez e o resultado (75 mil números) fica em
``<saida>/frac_invalido_ir.csv``; as execuções seguintes reaproveitam o cache em segundos.

FIDELIDADE AO NOTEBOOK
----------------------
As funções ``ler_info_tcir``, ``detectar_duplicatas``, ``carregar_tabela``, ``construir_janelas`` e ``dividir_folds`` são cópias das do
notebook (mesmas regras e mesmos parâmetros de ``Config``). Se mudar a regra no notebook, mude aqui também. A
definição de pixel inválido é a do notebook: NaN, infinito ou > 1000.

USO
---
    python3 estatisticas_janelas.py                       # usa ./data
    python3 estatisticas_janelas.py --dir-dados /caminho/data --saida resultados_janelas
    python3 estatisticas_janelas.py --refazer-varredura   # ignora o cache da fração de IR ausente

SAÍDAS (em ``resultados_janelas/``)
-----------------------------------
    estatisticas_janelas.json   todos os números (para conferir/citar)
    estatisticas_janelas.md     relatório legível
    frac_invalido_ir.csv        cache da varredura das imagens
"""
import argparse
import io
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- parâmetros (iguais ao Config do notebook)
CANAL_IR = 0                 # ordem dos canais no TCIR: 0=IR, 1=WV, 2=VIS, 3=PMW
RECORTE_ARMAZENADO = 92      # lado (px) do recorte central guardado antes da rotação
INVALIDO_ACIMA_DE = 1000.0   # pixels NaN ou > 1000 são inválidos (como no TCC)
MAX_FRAC_NAN_IR = 0.10       # quadro com mais de 10% de IR ausente é inválido
PASSO_HORAS = 3              # espaçamento regular entre quadros consecutivos
TAM_JANELA = 5               # nº de quadros da sequência (t-12h … t)
HORIZONTE_HORAS = 24         # Δ da intensificação rápida
LIMIAR_RI_NOS = 30.0         # RI = aumento de Vmax ≥ 30 nós em 24 h
TOL_DUPLICATA_GRAUS = 0.25   # IDs diferentes, mesmo instante e posição a ≤ isto (graus) = mesmo ciclone físico
N_FOLDS = 5
SEED_ESTRUTURAL = 2024
ANO_FIM_TREINO, ANO_FIM_VALIDACAO = 2014, 2016   # splits do TCC: treino 2003–2014, validação 2015–2016, teste 2017
NOMES_H5 = ("TCIR-ATLN_EPAC_WPAC-001.h5", "TCIR-CPAC_IO_SH-002.h5", "TCIR-ALL_2017-003.h5")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- leitura da tabela `info` (igual ao notebook)
class _UnpicklerRestrito(pickle.Unpickler):
    """Desserializa SOMENTE o necessário para recuperar um array NumPy de objetos (strings)."""
    _PERMITIDOS = {("numpy", "dtype"), ("numpy", "ndarray"),
                   ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct")}

    def find_class(self, module, name):
        if (module, name) not in self._PERMITIDOS:
            raise pickle.UnpicklingError(f"objeto não permitido no pickle: {module}.{name}")
        if module == "numpy.core.multiarray":  # NumPy 2 renomeou o módulo para numpy._core
            try:
                from numpy._core.multiarray import _reconstruct  # type: ignore
                return _reconstruct
            except Exception:
                pass
        return super().find_class(module, name)


def ler_info_tcir(caminho: Path) -> pd.DataFrame:
    """Lê a tabela `info` de um .h5 do TCIR (layout 'fixed' do pandas, lido direto com h5py)."""
    with h5py.File(caminho, "r") as f:
        g = f["info"]
        ordem = [c.decode() for c in g["axis0"][:]]
        indice = g["axis1"][:]
        dados, k = {}, 0
        while f"block{k}_items" in g:
            nomes = [c.decode() for c in g[f"block{k}_items"][:]]
            valores = g[f"block{k}_values"][:]
            if valores.dtype == object:
                valores = _UnpicklerRestrito(io.BytesIO(valores[0].tobytes())).load()
            for j, nome in enumerate(nomes):
                dados[nome] = valores[:, j]
            k += 1
    return pd.DataFrame(dados, index=indice)[ordem]


def localizar_arquivos(dir_dados: Path) -> dict:
    achados = {}
    # o arquivo mais próximo da raiz vence: data/TCIR-*.h5 (completo) antes de data/20percent/TCIR-*.h5 (subconjunto de teste)
    for p in sorted(dir_dados.rglob("*.h5"), key=lambda p: (len(p.relative_to(dir_dados).parts), str(p))):
        if "tcir" in p.name.lower() and p.name not in achados:
            achados[p.name] = p
    if not achados:
        raise FileNotFoundError(f"nenhum TCIR-*.h5 em {dir_dados}")
    return achados


def _distancia_graus(lon1, lat1, lon2, lat2):
    """Distância angular (graus) entre pontos da esfera — haversine; aceita vetores (broadcast)."""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def detectar_duplicatas(tabela: pd.DataFrame, tol_graus: float):
    """Devolve (id_ciclone, duplicata) alinhados a `tabela`: o ID unificado de cada linha e se ela deve ser removida.
    Só usa metadados (id_original, instante, lon, lat) — mesma regra do verificar_duplicatas.py."""
    ids = tabela["id_original"].to_numpy()
    lon = ((tabela["lon"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0   # CPAC usa 0..360
    lat = tabela["lat"].to_numpy(dtype=float)

    # Verificação B: pares de IDs diferentes, mesmo instante, posição a ≤ tol_graus → mesmo ciclone físico (union-find)
    pai: dict = {}

    def achar(x: str) -> str:
        pai.setdefault(x, x)
        while pai[x] != x:
            pai[x] = pai[pai[x]]            # compressão de caminho
            x = pai[x]
        return x

    for pos in tabela.groupby("instante").indices.values():     # poucos ciclones por instante: comparar todos os pares é barato
        if len(pos) < 2 or len(set(ids[pos])) < 2:
            continue
        d = _distancia_graus(lon[pos][:, None], lat[pos][:, None], lon[pos][None, :], lat[pos][None, :])
        for a, b in zip(*np.nonzero(np.triu(d <= tol_graus, k=1))):
            if ids[pos[a]] != ids[pos[b]]:
                ra, rb = achar(ids[pos[a]]), achar(ids[pos[b]])
                if ra != rb:
                    pai[rb] = ra

    # ID canônico de cada ciclone físico: o ID com mais registros (empate: o menor ID)
    n_por_id = tabela.groupby("id_original").size()
    raiz = tabela["id_original"].map(achar)
    canonico = {r: sorted(g, key=lambda x: (-n_por_id[x], x))[0]
                for r, g in tabela.assign(_raiz=raiz).groupby("_raiz")["id_original"].unique().items()}
    id_ciclone = raiz.map(canonico)

    # Verificações A e B: mais de uma linha para o mesmo (id_ciclone, instante) → fica uma só (a do ID com mais registros;
    # desempate: menor ID, arquivo, linha)
    t = pd.DataFrame({"id_ciclone": id_ciclone, "instante": tabela["instante"], "n_id": tabela["id_original"].map(n_por_id),
                      "id": tabela["id_original"], "arquivo": tabela["arquivo"], "linha": tabela["linha"]})
    ordem = t.sort_values(["id_ciclone", "instante", "n_id", "id", "arquivo", "linha"],
                          ascending=[True, True, False, True, True, True]).index
    rank = t.loc[ordem].groupby(["id_ciclone", "instante"]).cumcount()
    duplicata = pd.Series(False, index=tabela.index)
    duplicata.loc[ordem] = (rank > 0).to_numpy()
    return id_ciclone, duplicata


def carregar_tabela(arquivos: dict) -> pd.DataFrame:
    """Uma linha por imagem, com as duplicatas detectadas (mesma lógica do notebook)."""
    partes = []
    for nome, caminho in arquivos.items():
        d = ler_info_tcir(caminho)
        partes.append(pd.DataFrame({
            "arquivo": nome,
            "linha": np.arange(len(d)),
            "id_original": d["ID"].astype(str).to_numpy(),
            "bacia": d["data_set"].astype(str).to_numpy(),
            "instante": pd.to_datetime(d["time"].astype(str).to_numpy(), format="%Y%m%d%H"),
            "lon": d["lon"].to_numpy(dtype=float),
            "lat": d["lat"].to_numpy(dtype=float),
            "vmax": d["Vmax"].to_numpy(dtype=float),
        }))
    tabela = pd.concat(partes, ignore_index=True)
    tabela["ano"] = tabela["id_original"].str[:4].astype(int)

    tabela["id_ciclone"], tabela["duplicata"] = detectar_duplicatas(tabela, TOL_DUPLICATA_GRAUS)
    return tabela


# ----------------------------------------------------------------------------- varredura: fração de IR inválido por imagem
def varrer_frac_invalido_ir(tabela: pd.DataFrame, arquivos: dict, lote: int = 128) -> np.ndarray:
    """Fração de pixels inválidos (NaN/inf/>1000) do IR no recorte central 92×92 de cada imagem (alinhada a `tabela`)."""
    R = RECORTE_ARMAZENADO
    c0 = 100 - R // 2                 # o centro da imagem 201×201 é o pixel 100 → recorte [c0, c0+R)
    c1 = c0 + R
    frac = np.full(len(tabela), np.nan, dtype=np.float32)
    total, feitos, t0, ultimo = len(tabela), 0, time.time(), time.time()
    for nome, g in tabela.groupby("arquivo", sort=False):
        linhas, pos = g["linha"].to_numpy(), g.index.to_numpy()
        ordem = np.argsort(linhas)
        linhas, pos = linhas[ordem], pos[ordem]
        with h5py.File(arquivos[nome], "r") as f:
            m = f["matrix"]
            for a in range(0, len(linhas), lote):
                sel = linhas[a:a + lote]
                bloco = m[sel[0]:sel[-1] + 1, c0:c1]                          # (n, R, 201, 4): lê só as linhas do recorte
                ir = bloco[:, :, c0:c1, CANAL_IR][sel - sel[0]]               # (n, R, R)
                invalido = ~np.isfinite(ir) | (ir > INVALIDO_ACIMA_DE)
                frac[pos[a:a + lote]] = invalido.mean(axis=(1, 2))
                feitos += len(sel)
                if time.time() - ultimo > 20:
                    ultimo = time.time()
                    vel = feitos / max(ultimo - t0, 1e-9)
                    log(f"  varredura: {feitos}/{total} imagens ({vel:.0f} img/s, faltam ≈ {(total - feitos) / vel / 60:.0f} min)")
    assert not np.isnan(frac).any(), "alguma imagem ficou sem varredura"
    return frac


def obter_frac_invalido_ir(tabela, arquivos, cache: Path, refazer: bool) -> np.ndarray:
    if cache.exists() and not refazer:
        c = pd.read_csv(cache, dtype={"arquivo": str})
        m = tabela[["arquivo", "linha"]].merge(c, on=["arquivo", "linha"], how="left")
        if len(m) == len(tabela) and m["frac_inv_ir"].notna().all():
            log(f"cache da varredura: {cache}")
            return m["frac_inv_ir"].to_numpy(np.float32)
        log("cache incompleto/incompatível — refazendo a varredura")
    log(f"varrendo {len(tabela)} imagens (lê ~48 GB; roda uma vez e fica em cache)...")
    frac = varrer_frac_invalido_ir(tabela, arquivos)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"arquivo": tabela["arquivo"], "linha": tabela["linha"], "frac_inv_ir": frac}).to_csv(cache, index=False)
    log(f"cache gravado: {cache}")
    return frac


# ----------------------------------------------------------------------------- janelas (igual ao notebook)
def construir_janelas(tabela: pd.DataFrame, frac_inv_ir: np.ndarray):
    t = tabela.loc[~tabela["duplicata"]].copy()
    t["frac_inv_ir"] = frac_inv_ir[t.index.to_numpy()]
    t["quadro_valido"] = t["frac_inv_ir"] <= MAX_FRAC_NAN_IR
    t = t.sort_values(["id_ciclone", "instante"]).reset_index(drop=True)
    assert not t.duplicated(["id_ciclone", "instante"]).any(), "há (ciclone, instante) repetido"

    passo = pd.to_timedelta(PASSO_HORAS, unit="h")
    v = t[t["quadro_valido"]].reset_index(drop=True)
    # "corrida" = trecho contínuo de um mesmo ciclone com quadros válidos exatamente a cada 3 h
    nova_corrida = (v["id_ciclone"] != v["id_ciclone"].shift()) | (v["instante"].diff() != passo)
    v["corrida"] = nova_corrida.cumsum()
    v["pos_corrida"] = v.groupby("corrida").cumcount()

    n_prev = TAM_JANELA + 1           # 4 anteriores da janela + 2 para DTA/SDTA = 6 → cada janela usa 7 quadros consecutivos
    alvo = np.flatnonzero((v["pos_corrida"] >= n_prev).to_numpy())

    # rótulo futuro: Vmax(id, t+Δ) na série COMPLETA (o quadro futuro pode até ser inválido; o rótulo vale)
    serie_vmax = t.set_index(["id_ciclone", "instante"])["vmax"]
    a = v.iloc[alvo]
    fut = pd.MultiIndex.from_arrays([a["id_ciclone"].to_numpy(),
                                     (a["instante"] + pd.to_timedelta(HORIZONTE_HORAS, unit="h")).to_numpy()])
    vmax_fut = serie_vmax.reindex(fut).to_numpy(dtype=float)

    jan = pd.DataFrame({
        "id_ciclone": a["id_ciclone"].to_numpy(),
        "instante": a["instante"].to_numpy(),
        "ano": a["ano"].to_numpy(),
        "bacia": a["bacia"].to_numpy(),
        "vmax": a["vmax"].to_numpy(dtype=np.float32),
        "dvmax": (vmax_fut - a["vmax"].to_numpy()).astype(np.float32),   # NaN quando não há rótulo em t+Δ
    })
    return jan, t


def dividir_folds(ids_grupo: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Atribui cada GRUPO (ciclone) a um fold, balanceando o nº de janelas. Devolve o fold de cada janela."""
    grupos, inverso, contagem = np.unique(ids_grupo, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    ordem = np.lexsort((rng.random(len(grupos)), -contagem))
    carga, fold_do_grupo = np.zeros(n_folds), np.empty(len(grupos), dtype=np.int64)
    for g in ordem:
        f = int(np.argmin(carga))
        fold_do_grupo[g], carga[f] = f, carga[f] + contagem[g]
    return fold_do_grupo[inverso]


# ----------------------------------------------------------------------------- estatísticas
def resumo_vmax(x: pd.Series) -> dict:
    return {"n": int(x.notna().sum()), "media": float(x.mean()), "desvio_amostral_ddof1": float(x.std(ddof=1)),
            "desvio_populacional_ddof0": float(x.std(ddof=0)), "minimo": float(x.min()), "q25": float(x.quantile(.25)),
            "mediana": float(x.median()), "q75": float(x.quantile(.75)), "maximo": float(x.max())}


# categorias de intensidade (Saffir-Simpson, em nós; "DT" = depressão tropical, "TS" = tempestade tropical)
CATEGORIAS = [("DT (< 34)", -np.inf, 34), ("TS (34–63)", 34, 64), ("Cat. 1 (64–82)", 64, 83), ("Cat. 2 (83–95)", 83, 96),
              ("Cat. 3 (96–112)", 96, 113), ("Cat. 4 (113–136)", 113, 137), ("Cat. 5 (≥ 137)", 137, np.inf)]


def calcular(tabela: pd.DataFrame, serie: pd.DataFrame, jan: pd.DataFrame) -> dict:
    com_d = jan["dvmax"].notna()
    ri = jan["dvmax"] >= LIMIAR_RI_NOS
    split = np.select([jan["ano"] <= ANO_FIM_TREINO, jan["ano"] <= ANO_FIM_VALIDACAO], ["treino", "validacao"], "teste")
    por_ciclone = jan.groupby("id_ciclone").size()
    ideal = int(sum(max(n - (TAM_JANELA + 1), 0) for n in serie.groupby("id_ciclone").size()))
    folds = dividir_folds(jan["id_ciclone"].to_numpy(), N_FOLDS, SEED_ESTRUTURAL)
    tf = jan.assign(fold=folds).groupby("fold").agg(ciclones=("id_ciclone", "nunique"), janelas=("vmax", "size"),
                                                    vmax_medio=("vmax", "mean"), vmax_desvio=("vmax", "std"))
    cat = pd.cut(jan["vmax"], [b[1] for b in CATEGORIAS] + [np.inf], right=False, labels=[b[0] for b in CATEGORIAS])
    cat_cont = cat.value_counts().reindex([b[0] for b in CATEGORIAS])
    return {
        "parametros": {"tam_janela": TAM_JANELA, "quadros_por_janela_incl_dta": TAM_JANELA + 2, "passo_horas": PASSO_HORAS,
                       "max_frac_nan_ir": MAX_FRAC_NAN_IR, "horizonte_horas": HORIZONTE_HORAS, "limiar_ri_nos": LIMIAR_RI_NOS,
                       "n_folds": N_FOLDS, "seed_estrutural": SEED_ESTRUTURAL},
        "imagens_no_tcir": int(len(tabela)),
        "ids_originais": int(tabela["id_original"].nunique()),
        "registros_duplicados_removidos": int(tabela["duplicata"].sum()),
        "imagens_apos_duplicatas": int(len(serie)),
        "ciclones_apos_unir_ids": int(serie["id_ciclone"].nunique()),
        "quadros_invalidos": int((~serie["quadro_valido"]).sum()),
        "janelas": int(len(jan)),
        "ciclones_com_janela": int(jan["id_ciclone"].nunique()),
        "janelas_ideais_sem_quadros_invalidos": ideal,
        "janelas_perdidas_por_quadros_invalidos_ou_lacunas": ideal - int(len(jan)),
        "janelas_com_dvmax": int(com_d.sum()),
        "janelas_ri": int(ri.sum()),
        "pct_ri_das_rotuladas": float(100 * ri.sum() / com_d.sum()),
        "vmax_janelas": resumo_vmax(jan["vmax"]),
        "dvmax_janelas": resumo_vmax(jan["dvmax"]),
        "vmax_por_split": {s: resumo_vmax(jan.loc[split == s, "vmax"]) for s in ("treino", "validacao", "teste")},
        "janelas_por_split": {s: int((split == s).sum()) for s in ("treino", "validacao", "teste")},
        "janelas_por_ciclone": {"mediana": float(por_ciclone.median()), "maximo": int(por_ciclone.max()),
                                "minimo": int(por_ciclone.min())},
        "janelas_por_categoria_vmax": {k: int(v) for k, v in cat_cont.items()},
        "folds": tf.round(2).reset_index().to_dict(orient="records"),
        "baseline_prever_a_media_rmse": float(jan["vmax"].std(ddof=0)),
    }


def mil(n: int) -> str:
    """12345 → "12.345" (separador de milhar em português)."""
    return f"{n:,}".replace(",", ".")


def relatorio_md(r: dict) -> str:
    v, d = r["vmax_janelas"], r["dvmax_janelas"]
    pc = r["janelas_por_categoria_vmax"]
    n = r["janelas"]
    linhas = [
        "# Estatísticas das janelas (gerado por `estatisticas_janelas.py`)", "",
        "| item | valor |", "|---|---|",
        f"| Imagens no TCIR | {mil(r['imagens_no_tcir'])} ({mil(r['ids_originais'])} IDs) |",
        f"| Após remover duplicatas e unir IDs | {mil(r['imagens_apos_duplicatas'])} imagens, {mil(r['ciclones_apos_unir_ids'])} ciclones |",
        f"| Quadros inválidos (IR com > 10% de pixels ausentes) | {mil(r['quadros_invalidos'])} |",
        f"| Janelas utilizáveis | {mil(r['janelas'])} ({mil(r['ciclones_com_janela'])} ciclones) |",
        f"| Janelas com rótulo de ΔVmax | {mil(r['janelas_com_dvmax'])} |",
        f"| Intensificação rápida (ΔVmax ≥ {r['parametros']['limiar_ri_nos']:.0f} nós em {r['parametros']['horizonte_horas']} h) | "
        f"{mil(r['janelas_ri'])} ({r['pct_ri_das_rotuladas']:.2f}% das rotuladas) |",
        f"| Vmax das janelas | média {v['media']:.1f} nós, desvio {v['desvio_amostral_ddof1']:.1f} |", "",
        "## Vmax(t) das janelas (nós)", "",
        "| estatística | valor |", "|---|---|",
        f"| n | {v['n']} |", f"| média | {v['media']:.2f} |",
        f"| desvio padrão amostral (ddof=1) | {v['desvio_amostral_ddof1']:.2f} |",
        f"| desvio padrão populacional (ddof=0) | {v['desvio_populacional_ddof0']:.2f} |",
        f"| mínimo / Q25 / mediana / Q75 / máximo | {v['minimo']:.0f} / {v['q25']:.0f} / {v['mediana']:.0f} / {v['q75']:.0f} / {v['maximo']:.0f} |", "",
        f"Um \"modelo\" que sempre prevê a média teria RMSE ≈ {r['baseline_prever_a_media_rmse']:.1f} nós: é a régua mínima para os modelos.", "",
        "## ΔVmax = Vmax(t+24h) − Vmax(t) (nós)", "",
        f"média {d['media']:.2f}, desvio {d['desvio_amostral_ddof1']:.2f} (n = {d['n']}).", "",
        "## Janelas por categoria de Vmax", "", "| categoria | janelas | % |", "|---|---|---|",
        *[f"| {k} | {c} | {100 * c / n:.1f} |" for k, c in pc.items()], "",
        "## Divisão do TCC (por ano)", "", "| split | janelas | Vmax médio | desvio |", "|---|---|---|---|",
        *[f"| {s} | {r['janelas_por_split'][s]} | {r['vmax_por_split'][s]['media']:.1f} | "
          f"{r['vmax_por_split'][s]['desvio_amostral_ddof1']:.1f} |" for s in ("treino", "validacao", "teste")], "",
        "## Folds da validação cruzada (agrupada por ciclone)", "", "| fold | ciclones | janelas | Vmax médio | desvio |", "|---|---|---|---|---|",
        *[f"| {f['fold']} | {f['ciclones']} | {f['janelas']} | {f['vmax_medio']:.1f} | {f['vmax_desvio']:.1f} |" for f in r["folds"]], "",
        f"Janelas por ciclone: mediana {r['janelas_por_ciclone']['mediana']:.0f}, máx {r['janelas_por_ciclone']['maximo']}. "
        f"Janelas que existiriam sem quadros inválidos/lacunas: {r['janelas_ideais_sem_quadros_invalidos']} "
        f"(perdidas: {r['janelas_perdidas_por_quadros_invalidos_ou_lacunas']}).", ""]
    return "\n".join(linhas)


def main() -> None:
    aqui = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir-dados", type=Path, default=aqui / "data")
    ap.add_argument("--saida", type=Path, default=aqui / "resultados_janelas")
    ap.add_argument("--refazer-varredura", action="store_true", help="ignora o cache da fração de IR inválido")
    a = ap.parse_args()

    arquivos = localizar_arquivos(a.dir_dados)
    log("arquivos TCIR: " + ", ".join(arquivos))
    tabela = carregar_tabela(arquivos)
    frac = obter_frac_invalido_ir(tabela, arquivos, a.saida / "frac_invalido_ir.csv", a.refazer_varredura)
    jan, serie = construir_janelas(tabela, frac)
    r = calcular(tabela, serie, jan)

    a.saida.mkdir(parents=True, exist_ok=True)
    (a.saida / "estatisticas_janelas.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.saida / "estatisticas_janelas.md").write_text(relatorio_md(r), encoding="utf-8")
    print("\n" + relatorio_md(r))
    log(f"gravado em {a.saida}/")


if __name__ == "__main__":
    sys.exit(main())
