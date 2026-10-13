#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
treinar_remoto.py — treino da pipeline (CNN encoder + LSTM + cabeças KAN multitarefa) em um computador remoto com GPU.

É a parte de TREINO do `pipeline_tc_kaggle.ipynb` (seções 1 a 6) num script de linha de comando, para rodar sem Jupyter, sem
Kaggle e sem ninguém olhando (SSH + tmux/nohup), por exemplo numa máquina com uma RTX 5090. Ele faz:

    dados TCIR (.h5) → duplicatas → quadros IR+PMW (cache) → janelas de 5 quadros → folds por ciclone
        → treina fold × seed × variante  →  grava cada execução em  <saida>/runs/*.npz  (e *.pt com os pesos)

O QUE NÃO FAZ (continua no notebook): tabelas, testes estatísticos, baselines, figuras e explicabilidade (seções 7 a 10).

COMO O RESULTADO É IGUAL AO DO NOTEBOOK
    As definições (Config, modelo, KAN, tensor, janelas, folds, treino, assinatura) são cópias LITERAIS das do notebook, e
    `verificar_equivalencia_treino.py` confere isso (compara as árvores sintáticas). As únicas diferenças, todas declaradas lá:
    o log também vai para um arquivo, e `salvar_execucao` grava de forma atômica (uma queda no meio da gravação não deixa
    arquivo truncado). Os hiperparâmetros padrão são os do notebook; mudar um (--epocas, --lote) muda a assinatura das
    execuções, de propósito: execuções com hiperparâmetros diferentes nunca se misturam.

PREPARO DA MÁQUINA (uma vez)
    python -m venv .venv && source .venv/bin/activate          # (Windows: .venv\\Scripts\\activate)
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install numpy pandas h5py
    * Python 3.11 ou mais novo (compila em 3.11; testado em 3.13). Não precisa de Jupyter, matplotlib nem scipy.
    * RTX série 50 (5090, 5080...) = arquitetura Blackwell (sm_120): exige PyTorch >= 2.7 COMPILADO COM CUDA 12.8 (o índice `cu128`
      acima) e driver NVIDIA recente (>= 570). Com um PyTorch mais antigo o treino falha com "no kernel image is available". O
      script confere isso no início, em segundos, e diz o que fazer.
    * Os 3 arquivos TCIR-*.h5 (≈ 48 GB) precisam estar na máquina (--dados). Não aponte para uma pasta que também contenha cópias
      reduzidas (como data/20percent/): o script procura recursivamente e, no modo completo, recusa um conjunto com menos
      imagens que o TCIR inteiro (75.079).

COMO RODAR
    1) Teste rápido (minutos; confere ambiente, GPU e dados; os números NÃO valem como resultado; saída em <saida>/teste/):
           python treinar_remoto.py --dados /caminho/TCIR --saida /caminho/saida_pipeline --modo teste
    2) Experimento completo (100 execuções = 5 folds × 5 seeds × 4 variantes), sem depender do terminal aberto:
           nohup python treinar_remoto.py --dados /caminho/TCIR --saida /caminho/saida_pipeline --modo completo > treino.out 2>&1 &
           tail -f /caminho/saida_pipeline/log_treino.txt
    3) Caiu (SSH, energia, Ctrl-C, kill)? Rode o MESMO comando de novo: as execuções já gravadas são puladas.
    * Mais de uma GPU: escolha com `CUDA_VISIBLE_DEVICES=1 python treinar_remoto.py ...`.
    * Se a GPU sobrar (veja `nvidia-smi`: o modelo é pequeno), dá para rodar dois processos com folds diferentes
      (`--folds 0 1` e `--folds 2 3 4`) na MESMA pasta de saída; só inicie o 2º depois que o 1º avisar "cache gravado".

SAÍDAS (em <saida>/)
    log_treino.txt         log completo (acrescenta a cada execução do script)
    config.json            todos os parâmetros usados
    ambiente_treino.json   máquina, GPU, versões, argumentos, duração, pico de memória da GPU
    cache/                 quadros extraídos (≈ 2,5 GB; reaproveitado nas próximas execuções)
    runs/                  UM .npz por execução (predições do fold de teste) e o .pt (pesos) da 1ª seed de cada fold

DEPOIS DO TREINO: ANÁLISE NO NOTEBOOK
    Copie <saida>/runs/ (≈ 130 MB; o cache não precisa) para onde vai rodar o notebook e aponte para essa pasta:
        - na própria máquina remota:  TCIR_DIR=... TCIR_SAIDA=<a mesma saída> TCIR_MODO=completo, e execute o notebook;
        - no Kaggle: suba a pasta como dataset, anexe-o e use cfg.dirs_retomada = ("/kaggle/input/<dataset>/saida_pipeline",).
    O notebook reconhece as execuções pela assinatura (hiperparâmetros + janelas), pula o treino e gera tabelas e figuras.

CÓDIGO DE SAÍDA   0 = plano completo | 3 = execuções pendentes (acabou o --max-horas) | 130 = interrompido (Ctrl-C/SIGTERM)
"""
import argparse
import hashlib
import io
import json
import math
import os
import pickle
import platform
import random
import shutil
import signal
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")   # leitura de pastas somente leitura / em rede: evita erro de trava de arquivo
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

IMAGENS_TCIR = 75079   # 47.381 + 23.118 + 4.580: o TCIR inteiro (usado só para recusar uma pasta de dados incompleta)

# Estado global: o notebook também o tem, e as funções copiadas dele leem estes nomes. É preenchido por main().
device: torch.device = torch.device("cpu")
usar_amp: bool = False
DIR_SAIDA: Path = Path("saida_pipeline")
ASSINATURA: str = ""
_ARQ_LOG: Optional[Path] = None
_AMBIENTE: dict = {}


def log(msg: str) -> None:
    """Imprime com carimbo de hora e também acrescenta ao `log_treino.txt` da pasta de saída (sobrevive à queda do SSH)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    if _ARQ_LOG is not None:
        with open(_ARQ_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


# =====================================================================================================================
# 1. Configuração (copiada do notebook, seção 1: os valores padrão são os do experimento)
# =====================================================================================================================
@dataclass
class Config:
    # ------------------------------------------------------------------ execução
    modo: str = "teste"                    # "completo" | "teste"
    dir_dados: Optional[str] = None        # pasta com os TCIR-*.h5 (None = procura sozinho)
    dir_saida: Optional[str] = None        # onde gravar tudo (None = /kaggle/working/saida_pipeline ou ./saida_pipeline)
    detectar_duplicatas: bool = True       # detecta e remove registros redundantes do TCIR (seção 2.2); False = não mexe nos dados
    tol_duplicata_graus: float = 0.25      # IDs diferentes, mesmo instante e posição a ≤ isto (graus) = mesmo ciclone físico
    dirs_retomada: Tuple[str, ...] = ()    # pastas de saídas anteriores (para reaproveitar execuções já concluídas)
    seed_estrutural: int = 2024            # semente da divisão em folds e das subamostras (NÃO muda entre seeds de treino)
    usar_cache: bool = True                # grava os quadros extraídos em disco (≈ 2,5 GB); desligue se faltar espaço
    permitir_cpu: bool = False             # o modo "completo" recusa rodar sem GPU (levaria dias); True ignora a trava
    max_horas_treino: Optional[float] = None  # orçamento de tempo para NOVOS treinos nesta sessão (horas). None = automático:
                                           # 10 h no Kaggle (a sessão é limitada a ~12 h) e sem limite fora dele; 0 = sem limite

    # ------------------------------------------------------------------ dados / pré-processamento
    canal_ir: int = 0                      # ordem dos canais no TCIR: 0=IR, 1=WV, 2=VIS, 3=PMW
    canal_pmw: int = 3
    recorte_armazenado: int = 92           # lado (px) do recorte central guardado ANTES da rotação (≥ 64·√2 ≈ 90,5)
    recorte_final: int = 64                # lado (px) do tensor que entra na rede: (4, 64, 64)
    invalido_acima_de: float = 1000.0      # pixels NaN ou > 1000 são considerados inválidos (como no TCC)
    max_frac_nan_ir: float = 0.10          # quadro com mais de 10% de IR ausente é "inválido" (descartado das janelas)
    tratamento_invalidos: str = "tcc"      # pixels NaN/>1000 dentro de quadros válidos: "tcc" = vira 0 antes do z-score (igual ao TCC);
                                           # "mascara" = vira a média (z=0) e DTA/SDTA ficam 0 nesses pixels
    passo_horas: int = 3                   # espaçamento regular entre quadros consecutivos
    tam_janela: int = 5                    # nº de quadros da sequência: t-12h, t-9h, t-6h, t-3h, t
    horizonte_horas: int = 24              # Δ da intensificação rápida (a confirmar)
    limiar_ri_nos: float = 30.0            # RI = aumento de Vmax ≥ 30 nós em 24 h (Kaplan & DeMaria, 2003)
    prob_rotacao: float = 0.55             # chance de girar uma amostra de TREINO (anotação manuscrita lida como 0,55; a confirmar)
    max_ciclones: Optional[int] = None     # limita nº de ciclones (só para testes rápidos)

    # ------------------------------------------------------------------ modelo
    dim_embedding: int = 128               # tamanho do vetor latente e(t_i) de cada quadro
    dim_oculta: int = 128                  # tamanho do estado oculto h(t) da LSTM
    canais_cnn: Tuple[int, ...] = (32, 64, 128, 256)  # 4 convoluções com stride 2 (estilo CNN-TC): 64→32→16→8→4
    dropout: float = 0.1
    kan_ocultas: Tuple[int, ...] = (16,)   # largura das camadas ocultas da cabeça KAN: [128 → 16 → 1]
    kan_grid: int = 5                      # nº de intervalos do grid das B-splines
    kan_ordem: int = 3                     # grau das B-splines (3 = cúbica)
    mlp_ocultas: Tuple[int, ...] = (64,)   # cabeça MLP usada na ablação "sem KAN"

    # ------------------------------------------------------------------ treino
    tam_lote: int = 128
    tam_lote_aval: int = 256               # lote usado só na avaliação/explicabilidade (menor se a GPU for pequena)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epocas_max: int = 25
    paciencia: int = 8                     # early stopping: épocas sem melhora na validação interna
    clip_grad: float = 1.0
    pesos_tarefas: Dict[str, float] = field(default_factory=lambda: {"vmax": 1.0, "dvmax": 0.5})
    frac_validacao_interna: float = 0.10   # fração dos ciclones de treino reservada p/ early stopping
    usar_amp: bool = True                  # precisão mista (só vale em GPU CUDA)
    log_por_epoca: int = 5                 # imprime uma linha a cada N épocas (0 = só o resumo de cada execução)

    # ------------------------------------------------------------------ protocolo de validação
    n_folds: int = 5                       # CV agrupada por ciclone (k = 5)
    folds_executar: Optional[Tuple[int, ...]] = None   # None = todos; ex.: (0, 1) roda só 2 folds
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)           # 5 seeds por fold (controlam a estocasticidade do treino)
    variantes: Tuple[str, ...] = ("cnn_lstm_kan", "cnn_lstm_mlp", "cnn_kan_1frame", "cnn_lstm_kan_vmax")
    variante_referencia: str = "cnn_lstm_kan"          # modelo proposto: os outros são comparados contra ele
    alfa: float = 0.05                     # nível de significância dos testes (antes da correção de Holm)
    bootstrap_reps: int = 2000             # réplicas do bootstrap por ciclone (teste principal, seção 7.3b)
    baseline_sem_imagens: bool = True      # compara a rede com previsores de ΔVmax SEM imagens (seção 7.2b)
    outlier_erro_nos: float = 40.0         # |Vmax previsto − real| (nós) a partir do qual a janela entra no diagnóstico da seção 7.5

    # ------------------------------------------------------------------ explicabilidade
    gradcam_amostras: int = 256            # nº de janelas para o mapa Grad-CAM médio
    spline_topk: int = 4                   # nº de variáveis latentes mostradas nas curvas spline


# Variantes de modelo que podem ser comparadas estatisticamente (todas usam o MESMO tronco CNN):
VARIANTES = {
    "cnn_lstm_kan": dict(
        descricao="PROPOSTA — CNN encoder compartilhado + LSTM + cabeças KAN (Vmax e ΔVmax)",
        temporal="lstm", cabeca="kan", tarefas=("vmax", "dvmax")),
    "cnn_lstm_mlp": dict(
        descricao="Ablação do KAN — igual à proposta, mas com cabeças MLP",
        temporal="lstm", cabeca="mlp", tarefas=("vmax", "dvmax")),
    "cnn_kan_1frame": dict(
        descricao="Ablação do LSTM — só o quadro atual (sem sequência), cabeças KAN",
        temporal="nenhum", cabeca="kan", tarefas=("vmax", "dvmax")),
    "cnn_lstm_kan_vmax": dict(
        descricao="Ablação da multitarefa — LSTM + KAN treinados só para Vmax",
        temporal="lstm", cabeca="kan", tarefas=("vmax",)),
}


def aplicar_modo_teste(cfg: Config) -> None:
    """Versão minúscula do experimento (a mesma do notebook): poucos ciclones, 2 épocas — só confere o encanamento."""
    cfg.max_ciclones = 36
    cfg.n_folds = 3
    cfg.folds_executar = (0, 1)
    cfg.seeds = (0, 1)
    cfg.epocas_max = 2
    cfg.paciencia = 2
    cfg.gradcam_amostras = 32
    cfg.tam_lote = 64
    cfg.log_por_epoca = 1
    cfg.bootstrap_reps = 200


# =====================================================================================================================
# 1.1 Utilidades (copiadas do notebook)
# =====================================================================================================================
def fixar_semente(seed: int) -> None:
    """Fixa as sementes de Python, NumPy e PyTorch (CPU e GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def formatar_tempo(segundos: float) -> str:
    h, resto = divmod(int(segundos), 3600)
    m, s = divmod(resto, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# =====================================================================================================================
# 2. Dados (TCIR): tabela `info`, duplicatas, quadros IR+PMW em cache, janelas e rótulos (notebook, seções 2.1 a 2.4)
# =====================================================================================================================
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
    """Lê a tabela `info` de um .h5 do TCIR. Layout 'fixed' do pandas: axis0 = nomes das colunas,
    axis1 = índice, e blocos blockK_items/blockK_values que agrupam as colunas por tipo."""
    with h5py.File(caminho, "r") as f:
        g = f["info"]
        ordem = [c.decode() for c in g["axis0"][:]]
        indice = g["axis1"][:]
        dados, k = {}, 0
        while f"block{k}_items" in g:
            nomes = [c.decode() for c in g[f"block{k}_items"][:]]
            valores = g[f"block{k}_values"][:]
            if valores.dtype == object:  # bloco de strings (bacia, ID, time): pickle dentro do HDF5
                valores = _UnpicklerRestrito(io.BytesIO(valores[0].tobytes())).load()
            for j, nome in enumerate(nomes):
                dados[nome] = valores[:, j]
            k += 1
    return pd.DataFrame(dados, index=indice)[ordem]


def localizar_arquivos_tcir(cfg: Config) -> List[Path]:
    """Acha os TCIR-*.h5 (na pasta configurada ou em /kaggle/input, ./data, ../data)."""
    raizes = [Path(cfg.dir_dados)] if cfg.dir_dados else [Path("/kaggle/input"), Path("data"), Path("../data"), Path(".")]
    for raiz in raizes:
        if raiz.exists():
            achados, vistos = [], set()
            for p in sorted(p for p in raiz.rglob("*.h5") if "tcir" in p.name.lower()):
                if p.name in vistos:  # o mesmo arquivo anexado por dois datasets: usa só o primeiro (senão tudo duplicaria)
                    log(f"ATENÇÃO: {p} repete o nome de um arquivo já encontrado — ignorado")
                    continue
                vistos.add(p.name); achados.append(p)
            if achados:
                log("arquivos TCIR: " + ", ".join(str(p) for p in achados))
                return achados
    raise FileNotFoundError("Nenhum arquivo TCIR-*.h5 encontrado. Anexe o dataset no Kaggle ou ajuste cfg.dir_dados.")


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
    pai: Dict[str, str] = {}

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


def conferir_pixels_duplicatas(tabela: pd.DataFrame, arquivos: Dict[str, Path]):
    """Informativo: para cada linha removida, lê a imagem dela e a da linha mantida (mesmo id_ciclone e instante) e diz se
    são idênticas bit a bit. Devolve (n_pares, n_identicos). Lê só algumas dezenas de imagens."""
    rem = tabela[tabela["duplicata"]]
    if rem.empty:
        return 0, 0
    mantidas = tabela[~tabela["duplicata"]].set_index(["id_ciclone", "instante"])
    abertos, n_ident = {}, 0
    try:
        for _, r in rem.iterrows():
            m = mantidas.loc[(r["id_ciclone"], r["instante"])]
            imgs = []
            for nome, linha in ((r["arquivo"], int(r["linha"])), (m["arquivo"], int(m["linha"]))):
                if nome not in abertos:
                    abertos[nome] = h5py.File(arquivos[nome], "r")
                imgs.append(abertos[nome]["matrix"][linha])
            n_ident += int(np.array_equal(imgs[0], imgs[1], equal_nan=True))
    finally:
        for f in abertos.values():
            f.close()
    return len(rem), n_ident


def carregar_tabela_bruta(cfg: Config):
    """Lê o `info` de todos os arquivos e devolve a tabela (1 linha por imagem) já com as duplicatas detectadas."""
    arquivos = localizar_arquivos_tcir(cfg)
    partes = []
    for arq in arquivos:
        d = ler_info_tcir(arq)
        partes.append(pd.DataFrame({
            "arquivo": arq.name,
            "linha": np.arange(len(d)),                      # posição da imagem dentro do arquivo
            "id_original": d["ID"].astype(str).to_numpy(),
            "bacia": d["data_set"].astype(str).to_numpy(),
            "instante": pd.to_datetime(d["time"].astype(str).to_numpy(), format="%Y%m%d%H"),
            "lon": d["lon"].to_numpy(dtype=float),
            "lat": d["lat"].to_numpy(dtype=float),
            "vmax": d["Vmax"].to_numpy(dtype=float),         # nós
        }))
    tabela = pd.concat(partes, ignore_index=True)
    tabela["ano"] = tabela["id_original"].str[:4].astype(int)

    # ---- duplicatas: detectadas aqui mesmo, só com os metadados (seção 2.2)
    tabela["id_ciclone"] = tabela["id_original"]
    tabela["duplicata"] = False
    resumo_dup = {"detectadas": bool(cfg.detectar_duplicatas), "registros_removidos": 0, "ids_unificados": 0,
                  "pares_conferidos_nos_pixels": 0, "pares_identicos_bit_a_bit": 0}
    if cfg.detectar_duplicatas:
        tabela["id_ciclone"], tabela["duplicata"] = detectar_duplicatas(tabela, cfg.tol_duplicata_graus)
        n_uni = int((tabela["id_ciclone"] != tabela["id_original"]).groupby(tabela["id_original"]).any().sum())
        resumo_dup.update(registros_removidos=int(tabela["duplicata"].sum()), ids_unificados=n_uni)
        log(f"duplicatas: {resumo_dup['registros_removidos']} registros a remover, {n_uni} ID(s) unificado(s) com outro "
            f"(tolerância de posição {cfg.tol_duplicata_graus}°)")
        try:  # a conferência dos pixels é só informativa: se falhar, a remoção (feita pelos metadados) continua valendo
            n_pares, n_ident = conferir_pixels_duplicatas(tabela, {a.name: a for a in arquivos})
            resumo_dup.update(pares_conferidos_nos_pixels=n_pares, pares_identicos_bit_a_bit=n_ident)
            log(f"  conferência de pixels: {n_ident} de {n_pares} registros removidos são idênticos bit a bit à cópia mantida")
        except Exception as erro:
            log(f"  (conferência de pixels ignorada: {erro})")
    else:
        log("ATENÇÃO: cfg.detectar_duplicatas = False — seguindo SEM remover duplicatas (risco de vazamento na validação cruzada).")

    # ---- modo teste: subconjunto pequeno de ciclones (sempre inclui os ciclones duplicados, p/ exercitar a lógica)
    if cfg.max_ciclones:
        rng = np.random.default_rng(cfg.seed_estrutural)
        ids = np.array(sorted(tabela["id_ciclone"].unique()))
        obrigatorios = set(tabela.loc[tabela["id_ciclone"] != tabela["id_original"], "id_ciclone"])
        obrigatorios |= set(tabela.loc[tabela["duplicata"], "id_ciclone"])
        restantes = [i for i in ids if i not in obrigatorios]
        escolhidos = set(rng.choice(restantes, size=max(cfg.max_ciclones - len(obrigatorios), 1), replace=False)) | obrigatorios
        tabela = tabela[tabela["id_ciclone"].isin(escolhidos)].reset_index(drop=True)
        log(f"modo teste: {len(escolhidos)} ciclones, {len(tabela)} imagens")

    tabela["pos_frame"] = np.arange(len(tabela))  # posição da imagem no array de quadros extraído
    return tabela, {a.name: a for a in arquivos}, resumo_dup


def _blocos_contiguos(v: np.ndarray):
    """[(ini, fim), ...] com os trechos de índices consecutivos de um vetor crescente (leitura sequencial é mais rápida)."""
    if len(v) == 0:
        return []
    quebras = np.flatnonzero(np.diff(v) != 1) + 1
    inis = np.r_[0, quebras]
    fins = np.r_[quebras, len(v)]
    return list(zip(inis, fins))


def extrair_quadros(tabela: pd.DataFrame, arquivos: Dict[str, Path], cfg: Config, lote: int = 256):
    """Lê do HDF5 só IR e PMW no recorte central e devolve (quadros float16 (N,2,R,R) com NaN nos pixels inválidos,
    fração de pixels inválidos do IR (N,))."""
    N, R = len(tabela), cfg.recorte_armazenado
    c0 = 100 - R // 2                      # o centro da imagem 201×201 é o pixel 100 → recorte [c0, c0+R)
    c1 = c0 + R
    quadros = np.empty((N, 2, R, R), dtype=np.float16)
    frac_inv_ir = np.empty(N, dtype=np.float32)
    t0, lidos, ultimo_log = time.time(), 0, time.time()
    for nome, g in tabela.groupby("arquivo", sort=False):
        linhas, pos = g["linha"].to_numpy(), g["pos_frame"].to_numpy()
        ordem = np.argsort(linhas)
        linhas, pos = linhas[ordem], pos[ordem]
        with h5py.File(arquivos[nome], "r") as f:
            m = f["matrix"]
            for ini, fim in _blocos_contiguos(linhas):
                for a in range(ini, fim, lote):
                    b = min(fim, a + lote)
                    # lê só as LINHAS do recorte (trecho contíguo no disco = rápido) e corta as colunas na memória
                    bloco = m[linhas[a]:linhas[b - 1] + 1, c0:c1]                         # (n, R, 201, 4)
                    x = bloco[:, :, c0:c1, :][..., [cfg.canal_ir, cfg.canal_pmw]]         # (n, R, R, 2)
                    x = np.moveaxis(x, -1, 1)                                             # (n, 2, R, R)
                    invalido = ~np.isfinite(x) | (x > cfg.invalido_acima_de)
                    frac_inv_ir[pos[a:b]] = invalido[:, 0].mean(axis=(1, 2))
                    quadros[pos[a:b]] = np.where(invalido, np.nan, x).astype(np.float16)   # inválido → NaN (tratado depois)
                    lidos += b - a
                    if time.time() - ultimo_log > 20:  # progresso a cada ~20 s
                        ultimo_log = time.time()
                        log(f"  extraindo quadros: {lidos}/{N} ({lidos / max(ultimo_log - t0, 1e-9):.0f} img/s)")
    return quadros, frac_inv_ir


def obter_quadros(tabela: pd.DataFrame, arquivos: Dict[str, Path], cfg: Config):
    """Usa o cache em disco se existir (também procura nas pastas de retomada); senão extrai e grava."""
    chave_txt = "|".join(tabela["arquivo"] + ":" + tabela["linha"].astype(str)) + \
                f"|v2|{cfg.recorte_armazenado}|{cfg.canal_ir}|{cfg.canal_pmw}|{cfg.invalido_acima_de}"
    chave = hashlib.md5(chave_txt.encode()).hexdigest()[:12]
    if cfg.usar_cache:
        for base in [DIR_SAIDA] + [Path(d) for d in cfg.dirs_retomada]:
            qf, ff = base / "cache" / f"quadros_{chave}.npy", base / "cache" / f"fracinv_{chave}.npy"
            if qf.exists() and ff.exists():
                log(f"cache de quadros encontrado: {qf}")
                return np.load(qf), np.load(ff)
    log(f"extraindo {len(tabela)} quadros do HDF5 (leitura única; depois fica em cache)...")
    quadros, frac = extrair_quadros(tabela, arquivos, cfg)
    if cfg.usar_cache:
        qf, ff = DIR_SAIDA / "cache" / f"quadros_{chave}.npy", DIR_SAIDA / "cache" / f"fracinv_{chave}.npy"
        try:
            np.save(qf, quadros)
            np.save(ff, frac)
            log(f"cache gravado ({quadros.nbytes / 1e9:.2f} GB)")
        except OSError as erro:  # ex.: disco cheio — o cache é só uma otimização, então segue sem ele
            for arq in (qf, ff):
                arq.unlink(missing_ok=True)
            log(f"ATENÇÃO: não foi possível gravar o cache ({erro}); seguindo sem cache (use cfg.usar_cache=False para silenciar)")
    return quadros, frac


def construir_janelas(tabela: pd.DataFrame, frac_inv_ir: np.ndarray, cfg: Config):
    t = tabela.loc[~tabela["duplicata"]].copy()
    t["frac_inv_ir"] = frac_inv_ir[t["pos_frame"].to_numpy()]
    t["quadro_valido"] = t["frac_inv_ir"] <= cfg.max_frac_nan_ir
    t = t.sort_values(["id_ciclone", "instante"]).reset_index(drop=True)
    # Guarda de sanidade (não é a verificação de duplicatas, que é feita à parte):
    assert not t.duplicated(["id_ciclone", "instante"]).any(), \
        "há (ciclone, instante) repetido: confira cfg.detectar_duplicatas (seção 2.2)"

    passo = pd.to_timedelta(cfg.passo_horas, unit="h")
    v = t[t["quadro_valido"]].reset_index(drop=True)
    # "corrida" = trecho contínuo de um mesmo ciclone com quadros válidos exatamente a cada `passo` horas
    nova_corrida = (v["id_ciclone"] != v["id_ciclone"].shift()) | (v["instante"].diff() != passo)
    v["corrida"] = nova_corrida.cumsum()
    v["pos_corrida"] = v.groupby("corrida").cumcount()

    n_prev = cfg.tam_janela + 1                       # quadros anteriores ao alvo: 4 da janela + 2 p/ DTA/SDTA = 6
    alvo = np.flatnonzero((v["pos_corrida"] >= n_prev).to_numpy())
    idx_v = alvo[:, None] + np.arange(-n_prev, 1)[None, :]            # (W, 7) posições dentro de `v`
    idx_quadros = v["pos_frame"].to_numpy()[idx_v]                    # (W, 7) posições no array de quadros

    # rótulo futuro: procura Vmax(id, t+Δ) na série COMPLETA (o quadro futuro pode até ser inválido; o rótulo vale)
    serie_vmax = t.set_index(["id_ciclone", "instante"])["vmax"]
    a = v.iloc[alvo]
    fut = pd.MultiIndex.from_arrays([a["id_ciclone"].to_numpy(),
                                     (a["instante"] + pd.to_timedelta(cfg.horizonte_horas, unit="h")).to_numpy()])
    vmax_fut = serie_vmax.reindex(fut).to_numpy(dtype=float)

    jan = pd.DataFrame({
        "id_ciclone": a["id_ciclone"].to_numpy(),
        "instante": a["instante"].to_numpy(),
        "ano": a["ano"].to_numpy(),
        "bacia": a["bacia"].to_numpy(),
        "vmax": a["vmax"].to_numpy(dtype=np.float32),
        "dvmax": (vmax_fut - a["vmax"].to_numpy()).astype(np.float32),   # NaN quando não há rótulo em t+Δ
    })
    return jan, idx_quadros.astype(np.int64), t


# =====================================================================================================================
# 3. Tensor de entrada (4, 64, 64) por quadro, montado na GPU a cada lote (notebook, seção 3)
# =====================================================================================================================
@dataclass
class Dados:
    """Tudo que o treino precisa, já no dispositivo (GPU)."""
    quadros: torch.Tensor      # (N, 2, R, R) float16 — IR e PMW limpos, recorte central R×R
    idx: torch.Tensor          # (W, 7) int64 — posições dos 7 quadros (t-18h … t) de cada janela
    idx_np: np.ndarray         # a mesma matriz na CPU (para montar listas de quadros sem ir e voltar da GPU)
    y_vmax: torch.Tensor       # (W,) float32 — Vmax(t) em nós
    y_dvmax: torch.Tensor      # (W,) float32 — Vmax(t+Δ) − Vmax(t); NaN onde não há rótulo futuro
    jan: pd.DataFrame          # metadados (ciclone, instante, ano, bacia, rótulos)


def montar_dados(quadros_np: np.ndarray, idx_quadros: np.ndarray, janelas: pd.DataFrame) -> Dados:
    return Dados(
        quadros=torch.from_numpy(quadros_np).to(device),
        idx=torch.from_numpy(idx_quadros).to(device),
        idx_np=idx_quadros,
        y_vmax=torch.from_numpy(janelas["vmax"].to_numpy(np.float32)).to(device),
        y_dvmax=torch.from_numpy(janelas["dvmax"].to_numpy(np.float32)).to(device),
        jan=janelas,
    )


def estatisticas_canais(dados: Dados, pos_quadros: np.ndarray, cfg: Config, n_max: int = 3000, seed: int = 0):
    """Média e desvio-padrão GLOBAIS de (IR, PMW) numa amostra dos quadros de treino.
    Política "tcc": os pixels inválidos entram como 0 (como no TCC). Política "mascara": são ignorados."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(pos_quadros, size=min(n_max, len(pos_quadros)), replace=False)
    x = dados.quadros[torch.as_tensor(sel, device=dados.quadros.device)].float()          # (n, 2, R, R)
    if cfg.tratamento_invalidos == "tcc":
        x = torch.nan_to_num(x, nan=0.0)
        return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3)).clamp_min(1e-6)                # tensores de forma (2,)
    valido = ~torch.isnan(x)
    n = valido.sum(dim=(0, 2, 3)).clamp_min(1)
    xz = torch.where(valido, x, torch.zeros_like(x))
    mu = xz.sum(dim=(0, 2, 3)) / n
    var = (torch.where(valido, (x - mu.view(1, 2, 1, 1)) ** 2, torch.zeros_like(x))).sum(dim=(0, 2, 3)) / n
    return mu, var.sqrt().clamp_min(1e-6)


def rotacionar_e_recortar(x: torch.Tensor, angulos_graus: torch.Tensor, lado_saida: int) -> torch.Tensor:
    """Gira cada amostra de `x` (B, C, H, W) em torno do centro pelo seu ângulo e recorta `lado_saida`×`lado_saida`.

    Usa uma transformação afim: `affine_grid` cria, para cada pixel de saída, a coordenada correspondente na imagem de
    entrada (já girada e encolhida pelo fator lado_saida/H = recorte); `grid_sample` interpola bilinearmente.
    Com ângulo 0 cada pixel de saída cai exatamente num pixel de entrada → recorte central exato, sem borrar.
    """
    B, C, H, W = x.shape
    ang = torch.deg2rad(angulos_graus.to(x.dtype))
    s = lado_saida / H                                   # fração do quadro que o recorte final cobre (64/92)
    cos, sen = torch.cos(ang) * s, torch.sin(ang) * s
    theta = torch.zeros(B, 2, 3, dtype=x.dtype, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sen
    theta[:, 1, 0], theta[:, 1, 1] = sen, cos
    grade = F.affine_grid(theta, size=(B, C, lado_saida, lado_saida), align_corners=False)
    return F.grid_sample(x, grade, mode="bilinear", padding_mode="zeros", align_corners=False)


def montar_lote(dados: Dados, ib: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor, cfg: Config,
                treino: bool) -> torch.Tensor:
    """Monta o tensor de entrada (B, 5, 4, 64, 64) das janelas de índices `ib`.

    canais: 0=IR(z), 1=PMW(z), 2=DTA, 3=SDTA — quadros em ordem cronológica (t-12h … t).
    """
    x = dados.quadros[dados.idx[ib]].float()                         # (B, 7, 2, R, R): t-18h … t (NaN = pixel inválido)
    if cfg.tratamento_invalidos == "tcc":
        x = torch.nan_to_num(x, nan=0.0)                              # TCC: NaN/>1000 → 0 ANTES do z-score
    z = (x - mu.view(1, 1, 2, 1, 1)) / sd.view(1, 1, 2, 1, 1)         # 1. z-score por canal
    ir, pmw = z[:, :, 0], z[:, :, 1]                                  # (B, 7, R, R)
    # 2./3. Diferenças temporais calculadas na grade ORIGINAL (antes da rotação), para os 5 quadros da janela
    #       (índices 2…6 dos 7 lidos); o quadro n-1 é o índice 1…5 e o n-2 é o índice 0…4.
    dta = (ir[:, 2:] - ir[:, 1:-1]).abs()
    sdta = (ir[:, 2:] - 2.0 * ir[:, 1:-1] + ir[:, :-2]).abs()
    t = torch.stack([ir[:, 2:], pmw[:, 2:], dta, sdta], dim=2)        # (B, 5, 4, R, R)
    # política "mascara": o NaN se propaga pelas contas (DTA/SDTA ficam NaN se algum quadro envolvido era inválido
    # naquele pixel); aqui viram 0 = valor médio. Na política "tcc" não há NaN e esta linha não faz nada.
    t = torch.nan_to_num(t, nan=0.0)
    B, T, C, R, _ = t.shape
    t = t.reshape(B, T * C, R, R)               # 5 quadros × 4 canais = 20 "canais": UMA rotação vale para todos
    lado = cfg.recorte_final
    if treino and cfg.prob_rotacao > 0:
        girar = torch.rand(B, device=t.device) < cfg.prob_rotacao     # só uma fração das amostras é girada
        ang = torch.rand(B, device=t.device) * 360.0 * girar          # ângulo 0 = sem rotação
        t = rotacionar_e_recortar(t, ang, lado)
    else:                                                             # validação/teste: sem aumento de dados
        m = (R - lado) // 2
        t = t[:, :, m:m + lado, m:m + lado]
    return t.reshape(B, T, C, lado, lado)


# =====================================================================================================================
# 4. Modelo: CNN encoder (pesos compartilhados) → LSTM → cabeças KAN/MLP (notebook, seção 4)
# =====================================================================================================================
class CodificadorCNN(nn.Module):
    """CNN encoder: (N, 4, 64, 64) → vetor latente (N, dim_embedding). Estilo CNN-TC (stride 2, sem pooling)."""

    def __init__(self, canais_entrada: int = 4, canais=(32, 64, 128, 256), dim_embedding: int = 128,
                 lado_entrada: int = 64, dropout: float = 0.1):
        super().__init__()
        blocos, c_in = [], canais_entrada
        for c in canais:
            blocos += [nn.Conv2d(c_in, c, kernel_size=3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(c), nn.ReLU()]
            c_in = c
        self.convs = nn.Sequential(*blocos)               # o último bloco (ReLU) é o alvo do Grad-CAM
        lado = lado_entrada // (2 ** len(canais))          # 64 → 4
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                nn.Linear(c_in * lado * lado, dim_embedding),
                                nn.LayerNorm(dim_embedding))      # embedding com escala ~1 → portões da LSTM não saturam

    def forward(self, x):
        return self.fc(self.convs(x))


class CamadaKAN(nn.Module):
    """Uma camada Kolmogorov-Arnold: y_j = Σ_i φ_{j,i}(x_i), com φ(x) = w_b·silu(x) + Σ_m c_m·B_m(x).

    Parâmetros: `peso_base` (n_out, n_in) e `coef` (n_out, n_in, grid+ordem) — uma curva por aresta (j, i).
    O grid é igual para todas as entradas: `grid` intervalos em `faixa`, estendido `ordem` nós para cada lado.
    """

    def __init__(self, n_in: int, n_out: int, grid: int = 5, ordem: int = 3, faixa=(-1.0, 1.0)):
        super().__init__()
        self.n_in, self.n_out, self.grid, self.ordem = n_in, n_out, grid, ordem
        h = (faixa[1] - faixa[0]) / grid
        nos = torch.arange(-ordem, grid + ordem + 1, dtype=torch.float32) * h + faixa[0]   # grid + 2·ordem + 1 nós
        self.register_buffer("nos", nos)
        self.peso_base = nn.Parameter(torch.empty(n_out, n_in))
        self.coef = nn.Parameter(torch.empty(n_out, n_in, grid + ordem))
        nn.init.kaiming_uniform_(self.peso_base, a=math.sqrt(5))          # mesma inicialização do nn.Linear
        nn.init.normal_(self.coef, mean=0.0, std=0.1 / math.sqrt(n_in))   # splines começam quase planas (função ≈ caminho base)

    def bases(self, x: torch.Tensor) -> torch.Tensor:
        """B-splines B_m(x) por Cox-de Boor. x: (..., n_in) → (..., n_in, grid+ordem)."""
        t = self.nos
        x = x.unsqueeze(-1)
        b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)               # grau 0: degraus (1 só no intervalo que contém x)
        for k in range(1, self.ordem + 1):                          # cada grau mistura dois vizinhos do grau anterior
            b = ((x - t[:-(k + 1)]) / (t[k:-1] - t[:-(k + 1)])) * b[..., :-1] \
                + ((t[k + 1:] - x) / (t[k + 1:] - t[1:-k])) * b[..., 1:]
        return b

    def forward(self, x: torch.Tensor) -> torch.Tensor:              # x: (B, n_in) → (B, n_out)
        base = F.linear(F.silu(x), self.peso_base)                   # Σ_i w_b[j,i]·silu(x_i)
        spline = F.linear(self.bases(x).flatten(1), self.coef.flatten(1))   # Σ_i Σ_m c[j,i,m]·B_m(x_i)
        return base + spline

    # ---- utilidades de explicabilidade (seção 9) ------------------------------------------------
    @torch.no_grad()
    def valores_arestas(self, x: torch.Tensor) -> torch.Tensor:
        """φ_{j,i}(x_i) de TODAS as arestas para cada amostra: (N, n_out, n_in)."""
        base = F.silu(x).unsqueeze(1) * self.peso_base.unsqueeze(0)
        spline = torch.einsum("nim,oim->noi", self.bases(x), self.coef)
        return base + spline

    @torch.no_grad()
    def curva(self, j: int, i: int, xs: torch.Tensor) -> torch.Tensor:
        """A função aprendida φ_{j,i} avaliada em uma grade 1-D `xs` (para desenhar a spline da aresta i→j)."""
        spline = self.bases(xs.view(-1, 1))[:, 0, :] @ self.coef[j, i]
        return F.silu(xs) * self.peso_base[j, i] + spline


class CabecaKAN(nn.Module):
    """Cabeça de saída feita de camadas KAN: [dim_in → ocultas… → 1]. Camadas mais fundas recebem entradas não limitadas,
    então usam um grid mais largo; a primeira recebe h(t) ∈ (−1, 1)."""

    def __init__(self, dim_in: int, ocultas=(16,), grid: int = 5, ordem: int = 3):
        super().__init__()
        dims = [dim_in, *ocultas, 1]
        self.camadas = nn.ModuleList(
            CamadaKAN(a, b, grid, ordem, faixa=(-1.0, 1.0) if i == 0 else (-3.0, 3.0))
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])))

    def forward(self, h):
        for camada in self.camadas:
            h = camada(h)
        return h


class CabecaMLP(nn.Module):
    """Cabeça MLP (ablação "sem KAN"): [dim_in → ocultas… → 1] com ReLU."""

    def __init__(self, dim_in: int, ocultas=(64,)):
        super().__init__()
        dims, camadas = [dim_in, *ocultas], []
        for a, b in zip(dims[:-1], dims[1:]):
            camadas += [nn.Linear(a, b), nn.ReLU()]
        camadas.append(nn.Linear(dims[-1], 1))
        self.rede = nn.Sequential(*camadas)

    def forward(self, h):
        return self.rede(h)


class ModeloTC(nn.Module):
    """CNN encoder (compartilhado) → LSTM → cabeças multitarefa.

    forward(x): x (B, T, 4, H, W) → ({tarefa: (B,)}, h(t) (B, dim_oculta))
    """

    def __init__(self, cfg: Config, variante: str):
        super().__init__()
        v = VARIANTES[variante]
        self.temporal, self.tarefas = v["temporal"], tuple(v["tarefas"])
        self.encoder = CodificadorCNN(4, cfg.canais_cnn, cfg.dim_embedding, cfg.recorte_final, cfg.dropout)
        if self.temporal == "lstm":
            self.lstm = nn.LSTM(cfg.dim_embedding, cfg.dim_oculta, num_layers=1, batch_first=True)
        else:  # sem sequência: projeta o embedding do quadro atual p/ o mesmo espaço (−1, 1) do estado da LSTM
            self.proj = nn.Linear(cfg.dim_embedding, cfg.dim_oculta)
        criar = (lambda: CabecaKAN(cfg.dim_oculta, cfg.kan_ocultas, cfg.kan_grid, cfg.kan_ordem)) \
            if v["cabeca"] == "kan" else (lambda: CabecaMLP(cfg.dim_oculta, cfg.mlp_ocultas))
        self.cabecas = nn.ModuleDict({t: criar() for t in self.tarefas})   # UMA cabeça independente por tarefa

    def forward(self, x: torch.Tensor):
        B, T = x.shape[:2]
        if self.temporal == "nenhum":
            x, T = x[:, -1:], 1                                           # só o quadro atual
        e = self.encoder(x.flatten(0, 1)).view(B, T, -1)                  # (B, T, D): MESMOS pesos em todos os quadros
        if self.temporal == "lstm":
            _, (h_n, _) = self.lstm(e)                                    # lê e(t-12h) … e(t) em ordem
            h = h_n[-1]                                                   # estado oculto final h(t)
        else:
            h = torch.tanh(self.proj(e[:, -1]))
        tipo = h.device.type if h.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=tipo, enabled=False):             # cabeças sempre em float32 (splines são sensíveis)
            h = h.float()
            saidas = {t: cab(h).squeeze(-1) for t, cab in self.cabecas.items()}
        return saidas, h


def contar_parametros(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# =====================================================================================================================
# 5. Folds, perda, métricas de regressão e o laço de treino de UMA execução (notebook, seção 5)
# =====================================================================================================================
def dividir_folds(ids_grupo: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Atribui cada GRUPO (ciclone) a um fold, balanceando o nº de janelas. Devolve o fold de cada janela."""
    grupos, inverso, contagem = np.unique(ids_grupo, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    ordem = np.lexsort((rng.random(len(grupos)), -contagem))      # maiores primeiro; empates em ordem aleatória
    carga, fold_do_grupo = np.zeros(n_folds), np.empty(len(grupos), dtype=np.int64)
    for g in ordem:
        f = int(np.argmin(carga))
        fold_do_grupo[g], carga[f] = f, carga[f] + contagem[g]
    return fold_do_grupo[inverso]


def separar_indices(janelas: pd.DataFrame, folds: np.ndarray, fold: int, cfg: Config):
    """(treino, validação interna, teste) como índices de janelas. A validação interna reserva ciclones inteiros do treino."""
    teste = np.flatnonzero(folds == fold)
    treino_total = np.flatnonzero(folds != fold)
    ids = np.unique(janelas["id_ciclone"].to_numpy()[treino_total])
    rng = np.random.default_rng(cfg.seed_estrutural + 1000 + fold)
    ids_val = rng.choice(ids, size=max(1, int(round(len(ids) * cfg.frac_validacao_interna))), replace=False)
    em_val = np.isin(janelas["id_ciclone"].to_numpy()[treino_total], ids_val)
    return treino_total[~em_val], treino_total[em_val], teste


def metricas_regressao(y: np.ndarray, p: np.ndarray) -> dict:
    ok = ~(np.isnan(y) | np.isnan(p))
    if not ok.any():
        return dict(rmse=np.nan, mae=np.nan, vies=np.nan, n=0)
    e = p[ok] - y[ok]
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), vies=float(np.mean(e)), n=int(ok.sum()))


def _autocast():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=usar_amp) if device.type == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)


def _criar_scaler(ativo: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=ativo)
    except Exception:  # PyTorch mais antigo
        return torch.cuda.amp.GradScaler(enabled=ativo)


@torch.no_grad()
def prever(modelo: ModeloTC, dados: Dados, idx: np.ndarray, mu, sd, cfg: Config, norm: dict):
    """Predições (em nós) para as janelas `idx`, sem aumento de dados. Devolve {tarefa: array}."""
    modelo.eval()
    brutas = {t: [] for t in modelo.tarefas}
    for i in range(0, len(idx), cfg.tam_lote_aval):
        ib = torch.as_tensor(idx[i:i + cfg.tam_lote_aval], device=device)
        with _autocast():
            saidas, _ = modelo(montar_lote(dados, ib, mu, sd, cfg, treino=False))
        for t in brutas:
            brutas[t].append(saidas[t].float().cpu())
    return {t: torch.cat(v).numpy() * norm[t][1] + norm[t][0] for t, v in brutas.items()}


def perda_multitarefa(saidas: dict, yv_n: torch.Tensor, yd_n: torch.Tensor, pesos: dict) -> torch.Tensor:
    perda = torch.zeros((), device=yv_n.device)
    if "vmax" in saidas:
        perda = perda + pesos["vmax"] * F.mse_loss(saidas["vmax"], yv_n)
    if "dvmax" in saidas:
        m = ~torch.isnan(yd_n)                                    # só janelas com rótulo em t+Δ
        if m.any():
            perda = perda + pesos["dvmax"] * F.mse_loss(saidas["dvmax"][m], yd_n[m])
    return perda


def treinar_execucao(variante: str, fold: int, seed: int, dados: Dados, folds: np.ndarray, cfg: Config) -> dict:
    """Treina uma variante em um fold com uma seed e devolve predições do fold de teste + histórico."""
    t0 = time.time()
    fixar_semente(100 * seed + fold)
    idx_tr, idx_val, idx_te = separar_indices(dados.jan, folds, fold, cfg)

    # --- estatísticas calculadas SÓ no treino do fold (nada do teste vaza para a normalização)
    mu, sd = estatisticas_canais(dados, np.unique(dados.idx_np[idx_tr]), cfg, seed=cfg.seed_estrutural + fold)
    ytr_v = dados.y_vmax[torch.as_tensor(idx_tr, device=device)]
    ytr_d = dados.y_dvmax[torch.as_tensor(idx_tr, device=device)]
    ytr_d = ytr_d[~torch.isnan(ytr_d)]
    norm = {"vmax": (float(ytr_v.mean()), float(ytr_v.std())),
            "dvmax": (float(ytr_d.mean()), float(ytr_d.std())) if len(ytr_d) > 1 else (0.0, 1.0)}
    yv_n = (dados.y_vmax - norm["vmax"][0]) / norm["vmax"][1]
    yd_n = (dados.y_dvmax - norm["dvmax"][0]) / norm["dvmax"][1]

    modelo = ModeloTC(cfg, variante).to(device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    passos = max(len(idx_tr) // cfg.tam_lote, 1)
    agenda = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epocas_max * passos, pct_start=0.15)
    scaler = _criar_scaler(usar_amp)
    pesos = {t: cfg.pesos_tarefas[t] for t in modelo.tarefas}

    idx_tr_t = torch.as_tensor(idx_tr, device=device)
    y_val = dados.jan["vmax"].to_numpy(np.float32)[idx_val]
    historico, melhor, melhor_epoca, estado, sem_melhora = [], np.inf, -1, None, 0
    for ep in range(cfg.epocas_max):
        modelo.train()
        te = time.time()
        perm = idx_tr_t[torch.randperm(len(idx_tr_t), device=device)]
        soma = torch.zeros((), device=device)                      # acumula na GPU (evita sincronizar a cada passo)
        for i in range(passos):                                   # descarta o último lote incompleto (BatchNorm)
            ib = perm[i * cfg.tam_lote:(i + 1) * cfg.tam_lote]
            x = montar_lote(dados, ib, mu, sd, cfg, treino=True)  # rotação aleatória só aqui
            with _autocast():
                saidas, _ = modelo(x)
            perda = perda_multitarefa(saidas, yv_n[ib], yd_n[ib], pesos)   # soma ponderada das perdas das cabeças
            opt.zero_grad(set_to_none=True)
            scaler.scale(perda).backward()    # GradScaler: escala a perda p/ os gradientes em float16 não "zerarem" (no-op sem CUDA)
            scaler.unscale_(opt)              # desfaz a escala antes de cortar o gradiente
            nn.utils.clip_grad_norm_(modelo.parameters(), cfg.clip_grad)   # corte de gradiente: evita explosão na LSTM (BPTT)
            scaler.step(opt); scaler.update()
            agenda.step()                     # OneCycle: a taxa de aprendizado muda a CADA passo (sobe e depois desce)
            soma += perda.detach()
        pv = prever(modelo, dados, idx_val, mu, sd, cfg, norm)
        rmse_val = metricas_regressao(y_val, pv["vmax"])["rmse"]
        historico.append(dict(epoca=ep, perda_treino=float(soma) / passos, rmse_val_vmax=rmse_val, lr=agenda.get_last_lr()[0],
                              segundos=time.time() - te))
        if cfg.log_por_epoca and (ep + 1) % cfg.log_por_epoca == 0:
            log(f"    época {ep + 1:2d}/{cfg.epocas_max} | perda {historico[-1]['perda_treino']:.4f} | RMSE val Vmax {rmse_val:.2f} nós | "
                f"{historico[-1]['segundos']:.1f}s")
        if rmse_val < melhor - 1e-4:
            melhor, melhor_epoca, sem_melhora = rmse_val, ep, 0
            estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= cfg.paciencia:
                break

    modelo.load_state_dict(estado)                                # volta aos melhores pesos
    pt = prever(modelo, dados, idx_te, mu, sd, cfg, norm)
    pv_final = prever(modelo, dados, idx_val, mu, sd, cfg, norm)     # validação interna com os MELHORES pesos (ajusta o limiar de RI)
    nan = np.full(len(idx_te), np.nan, dtype=np.float32)
    nan_val = np.full(len(idx_val), np.nan, dtype=np.float32)
    res = dict(
        idx_teste=idx_te, y_vmax=dados.jan["vmax"].to_numpy(np.float32)[idx_te], pred_vmax=pt["vmax"].astype(np.float32),
        y_dvmax=dados.jan["dvmax"].to_numpy(np.float32)[idx_te], pred_dvmax=pt.get("dvmax", nan).astype(np.float32),
        y_dvmax_val=dados.jan["dvmax"].to_numpy(np.float32)[idx_val], pred_dvmax_val=pv_final.get("dvmax", nan_val).astype(np.float32),
        meta=dict(variante=variante, fold=fold, seed=seed, melhor_epoca=melhor_epoca, n_epocas=len(historico),
                  rmse_val=melhor, segundos=time.time() - t0, parametros=contar_parametros(modelo),
                  historico=historico, norm=norm, n_treino=len(idx_tr), n_val=len(idx_val), n_teste=len(idx_te)),
        estado=estado, mu=mu.cpu(), sd=sd.cpu())
    return res


# =====================================================================================================================
# 6. Protocolo: folds × seeds × variantes, com gravação imediata e retomada (notebook, seção 6)
# =====================================================================================================================
# Campos que não mudam o resultado e portanto ficam fora da assinatura (mesma lista do notebook).
_CAMPOS_SO_DE_EXECUCAO = {"modo", "dir_dados", "dir_saida", "detectar_duplicatas", "tol_duplicata_graus", "dirs_retomada", "variantes",
                          "folds_executar", "seeds", "variante_referencia", "alfa", "gradcam_amostras", "spline_topk",
                          "bootstrap_reps", "baseline_sem_imagens", "outlier_erro_nos",
                          "max_horas_treino", "usar_cache", "permitir_cpu", "tam_lote_aval", "log_por_epoca"}


def assinatura_experimento(cfg: Config, janelas: pd.DataFrame) -> str:
    """Hash curto dos hiperparâmetros que mudam o resultado + do conjunto de janelas."""
    d = {k: v for k, v in asdict(cfg).items() if k not in _CAMPOS_SO_DE_EXECUCAO}
    d["janelas"] = hashlib.md5("|".join(janelas["id_ciclone"] + "@" + janelas["instante"].astype(str)).encode()).hexdigest()
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]


def nome_run(variante: str, fold: int, seed: int) -> str:
    return f"{variante}__f{fold}__s{seed}__{ASSINATURA}"


def salvar_execucao(res: dict, caminho: Path, guardar_modelo: bool) -> None:
    """Igual ao do notebook, mas ATÔMICO: grava em runs/.tmp/ e só então move para runs/. O .pt vai primeiro e o .npz por ÚLTIMO,
    porque é o .npz que marca a execução como concluída (o plano pula quem já o tem): uma queda no meio da gravação nunca deixa
    um .npz truncado nem um .npz sem o .pt que lhe corresponde. O conteúdo dos arquivos é idêntico ao do notebook."""
    tmp = caminho.parent / ".tmp"
    tmp.mkdir(exist_ok=True)
    if guardar_modelo:  # pesos para a explicabilidade (seção 8 do notebook): só da primeira seed de cada fold
        pt = caminho.with_suffix(".pt")
        torch.save(dict(estado=res["estado"], mu=res["mu"], sd=res["sd"], meta=res["meta"]), tmp / pt.name)
        os.replace(tmp / pt.name, pt)
    np.savez_compressed(tmp / caminho.name, idx_teste=res["idx_teste"], y_vmax=res["y_vmax"], pred_vmax=res["pred_vmax"],
                        y_dvmax=res["y_dvmax"], pred_dvmax=res["pred_dvmax"],
                        y_dvmax_val=res["y_dvmax_val"], pred_dvmax_val=res["pred_dvmax_val"],
                        meta=np.array(json.dumps(res["meta"], default=float)))
    os.replace(tmp / caminho.name, caminho)


def executar_protocolo(dados: Dados, folds: np.ndarray, cfg: Config) -> int:
    """Treina o que falta do plano (fold × seed × variante). Devolve quantas execuções ficaram pendentes (0 = completo)."""
    pasta = DIR_SAIDA / "runs"
    folds_lista = tuple(cfg.folds_executar) if cfg.folds_executar is not None else tuple(range(cfg.n_folds))
    plano = [(f, s, v) for f in folds_lista for s in cfg.seeds for v in cfg.variantes]

    # retomada: traz para a pasta de saída as execuções já concluídas em sessões anteriores
    for d in cfg.dirs_retomada:
        for arq in Path(d).glob(f"runs/*__{ASSINATURA}.*"):
            if not (pasta / arq.name).exists():
                shutil.copy(arq, pasta / arq.name)
    pendentes = [(f, s, v) for f, s, v in plano if not (pasta / (nome_run(v, f, s) + ".npz")).exists()]
    log(f"plano: {len(plano)} execuções ({len(folds_lista)} folds × {len(cfg.seeds)} seeds × {len(cfg.variantes)} variantes) | "
        f"já concluídas: {len(plano) - len(pendentes)} | pendentes: {len(pendentes)}")

    tempos, t_ini = [], time.time()
    orcamento = (cfg.max_horas_treino or 0) * 3600           # segundos; 0 = sem limite
    if orcamento:
        log(f"orçamento de tempo para novos treinos: {cfg.max_horas_treino:g} h (depois disso para e segue para as tabelas)")
    for f, s, v in pendentes:
        if orcamento and tempos and (time.time() - t_ini) + float(np.mean(tempos)) > orcamento:
            log(f"orçamento de tempo esgotado: {len(pendentes) - len(tempos)} execuções ficaram pendentes. Rode de novo (ou numa nova "
                f"sessão com cfg.dirs_retomada) para completá-las; as tabelas abaixo usam só as {len(plano) - len(pendentes) + len(tempos)} concluídas.")
            break
        caminho = pasta / (nome_run(v, f, s) + ".npz")
        res = treinar_execucao(v, f, s, dados, folds, cfg)
        salvar_execucao(res, caminho, guardar_modelo=(s == cfg.seeds[0]))
        tempos.append(res["meta"]["segundos"])
        mv = metricas_regressao(res["y_vmax"], res["pred_vmax"])
        md = metricas_regressao(res["y_dvmax"], res["pred_dvmax"])
        restante = np.mean(tempos) * (len(pendentes) - len(tempos))
        log(f"[{len(tempos)}/{len(pendentes)}] {v:18s} fold {f} seed {s} | Vmax RMSE {mv['rmse']:.2f} MAE {mv['mae']:.2f} nós"
            + (f" | ΔVmax RMSE {md['rmse']:.2f}" if md["n"] else "")
            + f" | melhor época {res['meta']['melhor_epoca'] + 1}/{res['meta']['n_epocas']} | {formatar_tempo(res['meta']['segundos'])}"
            + f" | faltam ≈ {formatar_tempo(restante)}")
        del res
        if device.type == "cuda":
            torch.cuda.empty_cache()
    n_pend = len(pendentes) - len(tempos)
    log("protocolo concluído." if n_pend == 0 else f"protocolo INCOMPLETO: {n_pend} execuções pendentes.")
    return n_pend


# =====================================================================================================================
# 7. Específico do script: linha de comando, dispositivo (RTX 5090), verificação do modelo e função principal
# =====================================================================================================================
def ler_argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Treina a pipeline CNN + LSTM + KAN (todas as execuções fold × seed × variante) e grava runs/*.npz. "
                    "Detalhes: docstring deste arquivo.",
        epilog="Exemplos:\n"
               "  python treinar_remoto.py --dados /dados/TCIR --saida /dados/saida_pipeline --modo teste\n"
               "  python treinar_remoto.py --dados /dados/TCIR --saida /dados/saida_pipeline --modo completo\n"
               "  python treinar_remoto.py --dados /dados/TCIR --saida /dados/saida_pipeline --modo completo --folds 0 1 --max-horas 8",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dados", default=os.environ.get("TCIR_DIR"), metavar="DIR",
                   help="pasta com os 3 TCIR-*.h5 (busca recursiva). Padrão: $TCIR_DIR. Obrigatório.")
    p.add_argument("--saida", default=os.environ.get("TCIR_SAIDA") or "saida_pipeline", metavar="DIR",
                   help="pasta de saída (cache/, runs/, log). Padrão: $TCIR_SAIDA ou ./saida_pipeline")
    p.add_argument("--modo", choices=("teste", "completo"), default=os.environ.get("TCIR_MODO", "teste"),
                   help="teste = versão minúscula (minutos) | completo = protocolo inteiro (horas). Padrão: teste ($TCIR_MODO)")
    p.add_argument("--dispositivo", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                   help="auto = CUDA, senão MPS (Mac), senão CPU. Padrão: auto")
    p.add_argument("--permitir-cpu", action="store_true", help="deixa o modo completo rodar sem GPU (levaria dias)")
    # --- sobrescritas (opcionais): só o que mudar o resultado altera a assinatura das execuções
    p.add_argument("--variantes", nargs="+", metavar="V", help=f"subconjunto de {list(VARIANTES)} (padrão: as 4)")
    p.add_argument("--folds", nargs="+", type=int, metavar="N", help="folds a executar (padrão: todos; no teste: 0 1)")
    p.add_argument("--seeds", nargs="+", type=int, metavar="N", help="seeds (padrão: 0 1 2 3 4; no teste: 0 1)")
    p.add_argument("--epocas", type=int, metavar="N", help="épocas máximas por treino (padrão do notebook: 25) — muda a assinatura")
    p.add_argument("--lote", type=int, metavar="N", help="tamanho do lote de treino (padrão do notebook: 128) — muda a assinatura")
    p.add_argument("--lote-aval", type=int, metavar="N", help="tamanho do lote de avaliação (padrão: 256); não muda o resultado")
    p.add_argument("--max-horas", type=float, metavar="H",
                   help="para de INICIAR novos treinos depois de H horas (padrão: sem limite); rodar de novo retoma")
    p.add_argument("--sem-cache", action="store_true", help="não grava o cache de quadros em disco (≈ 2,5 GB)")
    p.add_argument("--retomar-de", nargs="+", metavar="DIR", help="pastas de saídas anteriores a reaproveitar (runs/ e cache/)")
    args = p.parse_args(argv)
    if not args.dados:
        p.error("informe --dados (ou defina TCIR_DIR): a pasta com os 3 TCIR-*.h5")
    return args


def montar_config(args: argparse.Namespace) -> Config:
    cfg = Config(modo=args.modo, dir_dados=args.dados, dir_saida=args.saida)
    if cfg.modo == "teste":
        aplicar_modo_teste(cfg)
    # sobrescritas explícitas da linha de comando (valem também no modo teste)
    if args.variantes:
        cfg.variantes = tuple(args.variantes)
    if args.folds is not None:
        cfg.folds_executar = tuple(args.folds)
    if args.seeds is not None:
        cfg.seeds = tuple(args.seeds)
    if args.epocas is not None:
        cfg.epocas_max = args.epocas
    if args.lote is not None:
        cfg.tam_lote = args.lote
    if args.lote_aval is not None:
        cfg.tam_lote_aval = args.lote_aval
    if args.max_horas is not None:
        cfg.max_horas_treino = args.max_horas
    if args.sem_cache:
        cfg.usar_cache = False
    if args.retomar_de:
        cfg.dirs_retomada = tuple(args.retomar_de)
    if args.permitir_cpu:
        cfg.permitir_cpu = True
    # as mesmas conferências do notebook (menos a da variante de referência, que só interessa à análise)
    assert cfg.tratamento_invalidos in ("tcc", "mascara"), 'tratamento_invalidos: "tcc" ou "mascara"'
    assert all(v in VARIANTES for v in cfg.variantes), f"variantes válidas: {list(VARIANTES)}"
    assert cfg.recorte_armazenado >= int(math.ceil(cfg.recorte_final * math.sqrt(2))), \
        "o recorte armazenado precisa cobrir o recorte final em QUALQUER rotação (≥ lado·√2)"
    assert all(0 <= f < cfg.n_folds for f in (cfg.folds_executar or ())), f"--folds: valores de 0 a {cfg.n_folds - 1}"
    return cfg


def escolher_dispositivo(pedido: str) -> torch.device:
    """auto: GPU CUDA se existir, senão MPS (Mac), senão CPU (a mesma ordem do notebook)."""
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if pedido == "cuda" and not torch.cuda.is_available():
        sys.exit("--dispositivo cuda, mas o PyTorch não enxerga nenhuma GPU CUDA (driver NVIDIA ausente ou PyTorch sem CUDA?). "
                 "Confira `nvidia-smi` e `python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"`.")
    if pedido == "mps" and not mps_ok:
        sys.exit("--dispositivo mps, mas o MPS não está disponível nesta máquina.")
    if pedido == "cuda" or (pedido == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if pedido == "mps" or (pedido == "auto" and mps_ok):
        return torch.device("mps")
    return torch.device("cpu")


def descrever_gpu() -> dict:
    """Nome, capacidade de computação, memória e arquiteturas para as quais este PyTorch foi compilado."""
    prop = torch.cuda.get_device_properties(0)
    return dict(gpu=prop.name, capacidade=f"sm_{prop.major}{prop.minor}", vram_gb=round(prop.total_memory / 1e9, 1),
                arquiteturas_do_pytorch=torch.cuda.get_arch_list(), torch_cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), n_gpus_visiveis=torch.cuda.device_count())


def ajuda_gpu_nao_suportada(info: dict) -> str:
    return (f"A GPU {info['gpu']} ({info['capacidade']}) pode não ser suportada por este PyTorch "
            f"({torch.__version__}, CUDA {info['torch_cuda']}; compilado para: {' '.join(info['arquiteturas_do_pytorch']) or '—'}).\n"
            "  As RTX série 50 (Blackwell, sm_120) exigem PyTorch >= 2.7 compilado com CUDA 12.8 e driver NVIDIA >= 570. Instale com:\n"
            "      pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128")


def verificar_modelo(cfg: Config) -> None:
    """Teste de sanidade do modelo (seção 4.4 do notebook) feito AQUI, no dispositivo de treino, antes de gastar horas:
    (i) as B-splines somam 1, (ii) formas dos tensores, (iii) o gradiente chega ao encoder, e (iv) uma passada com a precisão
    mista (AMP) que o treino usa. Numa GPU nova, se o PyTorch não tem kernels para ela, o erro aparece aqui, em segundos."""
    _kan = CamadaKAN(3, 2).to(device)
    _x = torch.rand(1000, 3, device=device) * 1.98 - 0.99                # dentro do grid (−1, 1)
    assert torch.allclose(_kan.bases(_x).sum(-1), torch.ones(1000, 3, device=device), atol=1e-5), "B-splines devem somar 1"
    _va = _kan.valores_arestas(_x[:8])
    assert torch.allclose(_va.sum(-1), _kan(_x[:8]), atol=1e-5), "a saída do nó deve ser a soma das φ das arestas"
    assert torch.allclose(_kan.curva(0, 1, _x[:8, 1]), _va[:, 0, 1], atol=1e-5), "curva() deve coincidir com a aresta (0←1)"
    _xb = torch.randn(4, 5, 4, cfg.recorte_final, cfg.recorte_final, device=device)
    for nome in cfg.variantes:
        _m = ModeloTC(cfg, nome).to(device)
        _saidas, _h = _m(_xb)
        sum(s.sum() for s in _saidas.values()).backward()
        _g = _m.encoder.convs[0].weight.grad
        assert _g is not None and torch.isfinite(_g).all() and _g.abs().sum() > 0, "gradiente não chegou ao encoder"
        assert _h.abs().max() <= 1.0 + 1e-6, "h(t) deveria ficar em (−1, 1)"
        log(f"  modelo {nome:18s} {contar_parametros(_m):>9,d} parâmetros | tarefas: {'+'.join(_m.tarefas)}")
    _m = ModeloTC(cfg, cfg.variantes[0]).to(device)                      # passada com precisão mista, como no treino
    _scaler = _criar_scaler(usar_amp)
    with _autocast():
        _saidas, _ = _m(_xb)
    _scaler.scale(sum(s.float().sum() for s in _saidas.values())).backward()
    assert all(torch.isfinite(p.grad).all() for p in _m.parameters() if p.grad is not None), "gradiente não finito com AMP"
    if device.type == "cuda":
        torch.cuda.synchronize()


def gravar_ambiente(**extra) -> None:
    """Grava (e vai atualizando) `ambiente_treino.json`: onde e como o treino rodou."""
    _AMBIENTE.update(extra)
    (DIR_SAIDA / "ambiente_treino.json").write_text(json.dumps(_AMBIENTE, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _ao_receber_sigterm(signum, frame):
    raise KeyboardInterrupt  # trata o `kill`/encerramento do sistema como Ctrl-C: avisa e sai com as execuções já gravadas


def main(argv=None) -> int:
    global device, usar_amp, DIR_SAIDA, ASSINATURA, _ARQ_LOG
    for fluxo in (sys.stdout, sys.stderr):  # consoles que não são UTF-8 (Windows) quebrariam com "Δ", "→", "≥"
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = ler_argumentos(argv)
    cfg = montar_config(args)
    t_inicio = time.time()

    # --- dispositivo
    device = escolher_dispositivo(args.dispositivo)
    usar_amp = bool(cfg.usar_amp and device.type == "cuda")
    if cfg.modo == "completo" and device.type == "cpu" and not cfg.permitir_cpu:
        sys.exit("Modo 'completo' sem GPU: o protocolo inteiro levaria dias na CPU. Confira o driver e o PyTorch com CUDA "
                 "(veja a docstring); para rodar na CPU mesmo assim, use --permitir-cpu; para só testar, --modo teste.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # tamanhos fixos de entrada → cuDNN escolhe os kernels mais rápidos

    # --- pastas e log
    DIR_SAIDA = Path(cfg.dir_saida)
    if cfg.modo == "teste":
        DIR_SAIDA = DIR_SAIDA / "teste"  # não mistura o teste com a execução de verdade
    for sub in ("cache", "runs"):
        (DIR_SAIDA / sub).mkdir(parents=True, exist_ok=True)
    _ARQ_LOG = DIR_SAIDA / "log_treino.txt"
    log("=" * 100)
    log(f"treinar_remoto.py | modo = {cfg.modo} | saída em {DIR_SAIDA.resolve()}")
    log(f"python {sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__} | pandas {pd.__version__} | "
        f"h5py {h5py.__version__} | {platform.platform()}")
    info_gpu = {}
    if device.type == "cuda":
        info_gpu = descrever_gpu()
        log(f"GPU: {info_gpu['gpu']} ({info_gpu['capacidade']}) | {info_gpu['vram_gb']} GB | CUDA {info_gpu['torch_cuda']} | "
            f"cuDNN {info_gpu['cudnn']} | {info_gpu['n_gpus_visiveis']} GPU(s) visível(is); usando a 0")
        if info_gpu["capacidade"] not in info_gpu["arquiteturas_do_pytorch"]:
            log("ATENÇÃO: " + ajuda_gpu_nao_suportada(info_gpu))
    log(f"dispositivo: {device} | precisão mista (AMP): {usar_amp}")
    (DIR_SAIDA / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    gravar_ambiente(inicio=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(), sistema=platform.platform(),
                    python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                    h5py=h5py.__version__, dispositivo=str(device), amp=usar_amp, argv=sys.argv[1:], modo=cfg.modo, **info_gpu)
    log(f"variantes = {cfg.variantes} | folds = {cfg.folds_executar or tuple(range(cfg.n_folds))} | seeds = {cfg.seeds} | "
        f"épocas ≤ {cfg.epocas_max} | lote {cfg.tam_lote}")
    if cfg.modo == "teste":
        log("!" * 78 + "\n  MODO TESTE: poucos ciclones e 2 épocas — só confere se o ambiente funciona.\n"
            "  Os números NÃO valem como resultado. Para o experimento real: --modo completo.\n" + "!" * 78)

    # --- o modelo funciona neste dispositivo? (antes de ler 48 GB de dados)
    log("verificando o modelo no dispositivo...")
    try:
        verificar_modelo(cfg)
    except RuntimeError:
        if device.type == "cuda":
            log("ERRO ao usar a GPU: " + ajuda_gpu_nao_suportada(info_gpu))
        raise
    log("modelo ok.")

    # --- dados: tabela, duplicatas, quadros (cache), janelas
    tabela_bruta, arquivos_h5, resumo_duplicatas = carregar_tabela_bruta(cfg)
    log(f"{len(tabela_bruta)} imagens | {tabela_bruta['id_original'].nunique()} IDs originais → "
        f"{tabela_bruta['id_ciclone'].nunique()} ciclones | anos {tabela_bruta['ano'].min()}–{tabela_bruta['ano'].max()}")
    if cfg.modo == "completo" and len(tabela_bruta) != IMAGENS_TCIR:
        sys.exit(f"Modo completo, mas a pasta de dados tem {len(tabela_bruta)} imagens (o TCIR inteiro tem {IMAGENS_TCIR}). "
                 f"Arquivos usados: {', '.join(str(p) for p in arquivos_h5.values())}. Confira --dados: uma pasta com cópias reduzidas "
                 "(como data/20percent/) faz o script usar os arquivos errados.")
    quadros_np, frac_invalido_ir = obter_quadros(tabela_bruta, arquivos_h5, cfg)
    log(f"quadros: {quadros_np.shape} {quadros_np.dtype} | {quadros_np.nbytes / 1e9:.2f} GB")
    janelas, idx_quadros, tabela_serie = construir_janelas(tabela_bruta, frac_invalido_ir, cfg)
    n_inv = int((~tabela_serie["quadro_valido"]).sum())
    n_rotul, n_ri = int(janelas["dvmax"].notna().sum()), int((janelas["dvmax"] >= cfg.limiar_ri_nos).sum())
    log(f"quadros na série: {len(tabela_serie)} (inválidos: {n_inv} = {100 * n_inv / len(tabela_serie):.2f}%)")
    log(f"janelas: {len(janelas)} de {janelas['id_ciclone'].nunique()} ciclones | com rótulo ΔVmax: {n_rotul} | "
        f"RI (ΔVmax ≥ {cfg.limiar_ri_nos:.0f} nós): {n_ri} ({100 * n_ri / max(n_rotul, 1):.1f}% das rotuladas)")
    assert len(janelas) > 50, "poucas janelas para treinar — confira os dados/modo de teste"
    dados = montar_dados(quadros_np, idx_quadros, janelas)
    if device.type != "cpu":
        del quadros_np  # a cópia na memória da CPU não é mais necessária
    log(f"dados no dispositivo: quadros {tuple(dados.quadros.shape)} "
        f"({dados.quadros.element_size() * dados.quadros.numel() / 1e9:.2f} GB) | janelas {tuple(dados.idx.shape)}")

    # --- folds agrupados por ciclone (fixos: iguais para todas as variantes e seeds)
    folds_janela = dividir_folds(janelas["id_ciclone"].to_numpy(), cfg.n_folds, cfg.seed_estrutural)
    assert (janelas.assign(fold=folds_janela).groupby("id_ciclone")["fold"].nunique() == 1).all(), "ciclone dividido entre folds!"
    tab_folds = janelas.assign(fold=folds_janela).groupby("fold").agg(
        ciclones=("id_ciclone", "nunique"), janelas=("vmax", "size"), vmax_medio=("vmax", "mean"),
        com_dvmax=("dvmax", lambda s: int(s.notna().sum())), ri=("dvmax", lambda s: int((s >= cfg.limiar_ri_nos).sum()))).round(1)
    log("folds:\n" + tab_folds.to_string())

    ASSINATURA = assinatura_experimento(cfg, janelas)
    log(f"assinatura do experimento: {ASSINATURA} (nome das execuções em runs/; o notebook calcula a mesma)")
    gravar_ambiente(assinatura=ASSINATURA, imagens=int(len(tabela_bruta)), ciclones=int(janelas["id_ciclone"].nunique()),
                    janelas=int(len(janelas)), duplicatas=resumo_duplicatas)

    # --- treino
    signal.signal(signal.SIGTERM, _ao_receber_sigterm)
    status, n_pend = "concluido", 0
    try:
        n_pend = executar_protocolo(dados, folds_janela, cfg)
        status = "concluido" if n_pend == 0 else "incompleto"
    except KeyboardInterrupt:
        status = "interrompido"
        log("INTERRUPÇÃO recebida: o treino em andamento foi descartado; as execuções já gravadas ficam em runs/. "
            "Rode o mesmo comando de novo para retomar.")

    # --- fecho
    n_ok = len(list((DIR_SAIDA / "runs").glob(f"*__{ASSINATURA}.npz")))
    pico = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    gravar_ambiente(fim=time.strftime("%Y-%m-%d %H:%M:%S"), status=status, execucoes_gravadas=n_ok, execucoes_pendentes=int(n_pend),
                    duracao_s=round(time.time() - t_inicio, 1), pico_memoria_gpu_gb=None if pico is None else round(pico, 2))
    log(f"{status.upper()} | execuções gravadas com esta assinatura: {n_ok} | duração desta sessão: {formatar_tempo(time.time() - t_inicio)}"
        + (f" | pico de memória da GPU: {pico:.2f} GB" if pico is not None else ""))
    if status == "concluido":
        log(f"Próximo passo: levar {DIR_SAIDA / 'runs'} para onde roda o notebook (TCIR_SAIDA ou cfg.dirs_retomada) e executá-lo: "
            "ele reconhece as execuções, pula o treino e gera tabelas e figuras.")
    return {"concluido": 0, "incompleto": 3, "interrompido": 130}[status]


if __name__ == "__main__":
    sys.exit(main())
