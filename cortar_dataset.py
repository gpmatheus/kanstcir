#!/usr/bin/env python3
"""Corta os 3 arquivos TCIR (.h5) em uma fração (padrão 20%) para testes rápidos no Kaggle.

Mantém a estrutura idêntica à original (mesmos nomes de arquivo, grupo `info` no layout 'fixed' do pandas,
dataset `matrix` com shape (N, 201, 201, 4) float32); só diminui o número de imagens e as linhas de `info`.

A amostragem é por CICLONE (ID), não por imagem solta: assim as sequências temporais ficam inteiras e a pipeline
(janelas de 7 quadros consecutivos, CV agrupada por ciclone) continua exercitando todo o código. As linhas
selecionadas mantêm a ordem original. Cada arquivo recebe ~`frac` das suas imagens.

Uso:
    python cortar_dataset.py                       # data/ -> data/20percent/
    python cortar_dataset.py --frac 0.2 --seed 0 --compressao 4
    python cortar_dataset.py --compressao 0        # sem gzip (arquivos ocupam ~20% do original em disco)
"""
import argparse
import io
import pickle
from pathlib import Path

import h5py
import numpy as np

ARQUIVOS = ["TCIR-ALL_2017-003.h5", "TCIR-ATLN_EPAC_WPAC-001.h5", "TCIR-CPAC_IO_SH-002.h5"]


class _Unpickler(pickle.Unpickler):
    _OK = {("numpy", "dtype"), ("numpy", "ndarray"),
           ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct")}

    def find_class(self, module, name):
        if (module, name) not in self._OK:
            raise pickle.UnpicklingError(f"objeto não permitido no pickle: {module}.{name}")
        if module == "numpy.core.multiarray":
            try:
                from numpy._core.multiarray import _reconstruct
                return _reconstruct
            except Exception:
                pass
        return super().find_class(module, name)


def ids_do_arquivo(g: h5py.Group) -> np.ndarray:
    """Coluna `ID` (string) de cada linha de `info`."""
    k = 0
    while f"block{k}_items" in g:
        nomes = [c.decode() for c in g[f"block{k}_items"][:]]
        if "ID" in nomes:
            v = g[f"block{k}_values"][:]
            if v.dtype == object:
                v = _Unpickler(io.BytesIO(v[0].tobytes())).load()
            return np.asarray(v[:, nomes.index("ID")]).astype(str)
        k += 1
    raise KeyError("coluna ID não encontrada em info")


def escolher_linhas(ids: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Sorteia ciclones inteiros até somar ~frac das linhas; devolve os índices (ordenados) das linhas."""
    unicos, inv, cont = np.unique(ids, return_inverse=True, return_counts=True)
    alvo = round(frac * len(ids))
    escolhidos, soma = [], 0
    for i in rng.permutation(len(unicos)):
        # pára no ponto mais próximo do alvo (não estoura por causa de um ciclone longo)
        if soma >= alvo or abs(soma + cont[i] - alvo) > abs(soma - alvo) and soma > 0:
            break
        escolhidos.append(i); soma += cont[i]
    marcado = np.zeros(len(unicos), bool); marcado[escolhidos] = True
    return np.flatnonzero(marcado[inv])


def cortar(origem: Path, destino: Path, frac: float, seed: int, compressao: int, bloco: int = 256):
    rng = np.random.default_rng(seed)
    with h5py.File(origem, "r") as src:
        g = src["info"]
        ids = ids_do_arquivo(g)
        sel = escolher_linhas(ids, frac, rng)
        n_total, n = len(ids), len(sel)
        print(f"{origem.name}: {n}/{n_total} imagens ({n / n_total:.1%}), "
              f"{len(np.unique(ids[sel]))}/{len(np.unique(ids))} ciclones", flush=True)

        with h5py.File(destino, "w") as dst:
            # ---- info: copia dataset a dataset, fatiando as linhas ----
            gd = dst.create_group("info")
            for k, v in g.attrs.items():
                gd.attrs[k] = v
            for nome, ds in g.items():
                if nome == "axis1" or nome.endswith("_values"):
                    if ds.dtype == object:  # bloco de strings: pickle de um array (N, ncols) dentro de um VLArray
                        arr = _Unpickler(io.BytesIO(ds[0].tobytes())).load()[sel]
                        blob = np.frombuffer(pickle.dumps(arr, protocol=4), dtype=np.uint8)
                        novo = gd.create_dataset(nome, shape=(1,), maxshape=(None,), dtype=ds.dtype, chunks=ds.chunks)
                        novo[0] = blob
                    else:
                        novo = gd.create_dataset(nome, data=ds[:][sel])
                else:
                    novo = gd.create_dataset(nome, data=ds[:])
                for k, v in ds.attrs.items():
                    novo.attrs[k] = v

            # ---- matrix: mesma dtype/shape além do eixo 0; copia em blocos (cabe em 9 GB de RAM) ----
            m = src["matrix"]
            kw = {}
            if compressao > 0:
                kw = dict(chunks=(1,) + m.shape[1:], compression="gzip", compression_opts=compressao, shuffle=True)
            md = dst.create_dataset("matrix", shape=(n,) + m.shape[1:], dtype=m.dtype, **kw)
            for i in range(0, n, bloco):
                idx = sel[i:i + bloco]
                # leitura por fatias contíguas (h5py exige índices crescentes; já estão)
                md[i:i + len(idx)] = np.stack([m[j] for j in idx])
                if (i // bloco) % 10 == 0:
                    print(f"  {min(i + bloco, n)}/{n}", flush=True)
    print(f"  -> {destino} ({destino.stat().st_size / 1e9:.2f} GB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origem", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--destino", type=Path, default=None, help="padrão: <origem>/20percent")
    ap.add_argument("--frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compressao", type=int, default=4, help="gzip 0-9 (0 = sem compressão)")
    a = ap.parse_args()
    destino = a.destino or a.origem / "20percent"
    destino.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS:
        cortar(a.origem / nome, destino / nome, a.frac, a.seed, a.compressao)


if __name__ == "__main__":
    main()
