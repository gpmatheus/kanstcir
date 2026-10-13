# Projeto: intensidade de ciclones tropicais com CNN encoder + LSTM + KAN multitarefa (TCIR)

> Contexto persistente do projeto para o Claude Code. **Estado em 2026-10-12.** Este arquivo é **autocontido**: reúne a especificação, os conceitos, as decisões, os resultados e os próximos passos, sem depender de nenhum outro documento. Leia as seções 1 a 3 primeiro; o restante é material de consulta.

## 1. Resumo

- **Quem:** pós-graduando em Computação (UNIPAMPA). O trabalho evolui o TCC do usuário (*Predição da velocidade de ciclones tropicais utilizando redes neurais convolucionais de baixa e alta complexidade com explicabilidade baseada em SHAP*, `versão final.pdf`) para um **artigo**.
- **Objetivo:** estimar a intensidade (Vmax) de ciclones tropicais e prever a intensificação rápida (ΔVmax) a partir de **sequências** de imagens de satélite do dataset TCIR, com a arquitetura **CNN encoder (pesos compartilhados) → LSTM → cabeças KAN multitarefa**. A contribuição central do artigo é essa combinação (KAN é de 2024; seu uso em regressão multitarefa de ciclones é novo), mais um protocolo estatístico rigoroso e explicabilidade nativa (Grad-CAM + splines da KAN).
- **Estado:** pipeline implementada num notebook PyTorch para o Kaggle e **rodada uma vez** (75 treinos). A análise desses resultados levou a ajustes no notebook que **ainda não foram executados em GPU nem com os dados completos**. Próximo passo: rodar a versão ajustada no Kaggle e analisar (seção 2).
- **Idioma:** português (BR) em tudo: conversa, comentários de código, explicações. O usuário precisa entender e justificar cada etapa, então explique o porquê e os conceitos, não só o código.
- Siglas e termos: glossário na seção 16.

## 2. O que fazer agora (em ordem)

1. **Rodar a pipeline ajustada no Kaggle** (será também o primeiro teste em GPU e em escala): importar `pipeline_tc_kaggle.ipynb`, anexar o dataset com os 3 `TCIR-*.h5`, acelerador **T4** (a P100 pode falhar com PyTorch novo), internet desligada (não há `pip install`). Primeiro `MODO = "teste"` (minutos), depois `MODO = "completo"` → *Save & Run All*. São ≈ 100 treinos (≈ 6–6,5 h; o orçamento padrão é 10 h; é retomável). Conferir no log: `duplicatas: 72 registros a remover, 2 ID(s) unificado(s) com outro` e 65.670 janelas / 1.373 ciclones. **Alternativa para o treino:** numa máquina própria com GPU (RTX 5090), `treinar_remoto.py` roda só o treino (seção 14.8); o notebook, apontado para o `runs/` dele, pula o treino e faz a análise (o passo 2 muda: o zip é o do notebook de análise).
2. **Baixar** o `resultados_pipeline.zip` (tem `runs/*.npz`, tabelas e figuras) **e** o notebook executado (`.ipynb` com o log).
3. **Ler nesta ordem:** `tabelas/bootstrap_por_ciclone.csv` (o que é significativo) → `tabelas/dvmax_modelos_vs_baselines.csv` (a rede bate o baseline sem imagens? e no RI?) → `tabelas/janelas_erro_grande_vmax.csv` + `figuras/09_*` → `tabelas/kan_latentes_correlacoes_*.csv`.
4. **Investigar a faixa horizontal** do gráfico predito × real (janelas com Vmax real entre 40 e 140 nós previstas em ≈ 25 nós pelos 3 modelos). Hipótese, não confirmada: ciclones sobre a **linha de data** (|lon| > 175°, p. ex. `201407E`, ago/2014), cujo recorte de 201×201 px cruza a emenda ±180° da grade (seção 9.4, item 2). Se confirmar, corrigir o recorte ou descartar essas janelas.
5. **Decidir com o usuário** os itens em aberto (seção 12), principalmente: o que fazer com ΔVmax (não superou um baseline sem imagens), RI como classificação, esparsificação da KAN e quais ablações entram no artigo.
6. Depois: escrever resultados e limitações; opcionalmente rodar um experimento com o split por ano do TCC (comparação estrita) e tratar o sobreajuste.

Ao reportar: números do artigo vêm da **nova** execução (a 1ª rodou sem remover duplicatas: 65.730 janelas, não 65.670); números do modo teste (CPU, dados reduzidos) **não são resultado**.

## 3. Regras de trabalho

- **Todo cálculo cujo número aparece em entregas fica salvo como `.py` na raiz do projeto**, com saída em pasta própria (padrão: `verificar_duplicatas.py` → `resultados_duplicatas/`; `estatisticas_janelas.py` → `resultados_janelas/`). Nunca deixar cálculo só em heredoc ou script temporário (já aconteceu: os números de média e desvio do Vmax se perderam e foram recriados).
- Notebook **autocontido**: sem `pip install`, sem internet, retomável. Mudou algo nele: compilar as células com Python 3.11 (versão do Kaggle), rodar o modo teste e registrar a mudança neste arquivo (seções 10 e 11).
- **`treinar_remoto.py` contém cópias literais das definições de treino do notebook** (config, janelas, tensor, modelo, KAN, folds, treino, assinatura). Mudou qualquer uma delas no notebook (ou no script): mudar nos dois lados e rodar `python3 -I verificar_equivalencia_treino.py` (compara as árvores sintáticas; sai com código 1 se algo diverge). Sem isso o script treina outra coisa e as execuções dele deixam de ser reconhecidas pelo notebook.
- Ler notebooks baixados do Kaggle com `python3 -I`, como JSON; nunca executá-los.
- **Não alterar o modelo proposto** (entradas, cabeças, KAN) sem decisão explícita do usuário. Os itens da seção 12 são decisões do usuário: apresente a recomendação e o custo; não decida sozinho.
- Reportar com fidelidade o que foi testado e o que não foi. Números do modo teste não são resultado do experimento.
- Antes de citar qualquer conclusão em texto do artigo, reler a seção 9.3: "KAN melhor que MLP" e "a rede supera o baseline sem imagens em ΔVmax/RI" **não** se sustentam com os dados da 1ª execução.
- Este arquivo é a fonte única de contexto: resultados e decisões novos entram nas seções correspondentes (resultados na 9, ajustes na 10, decisões nas 11 e 12).

## 4. Mapa do repositório e entregáveis

| Arquivo | O que é |
|---|---|
| `pipeline_tc_kaggle.ipynb` | Pipeline completa (PyTorch; 60 células, 30 de código): dados → tensor → CNN+LSTM+KAN → CV agrupada × seeds → baselines sem imagens → bootstrap por ciclone + Wilcoxon + Holm → diagnóstico de janelas → Grad-CAM e splines. Autocontida (sem `pip install`, sem internet), acha os dados sozinha, usa GPU se houver, grava cada treino ao terminar e **retoma** se a sessão cair. |
| `verificar_duplicatas.py` | Auditoria independente (pedido explícito do usuário): as verificações A–D da seção 7.3, inclusive hash de todas as imagens (`--varrer-imagens`). Saída em `resultados_duplicatas/` (relatório, resumo em JSON e `duplicatas_tcir.csv` com a decisão manter/remover por registro). O notebook **não** depende dele nem do CSV. |
| `estatisticas_janelas.py` | Reproduz as regras de janelas do notebook (sem PyTorch; as funções são cópias das do notebook: se a regra mudar lá, mudar aqui) e calcula nº de janelas, RI, média/dp do Vmax, splits e folds. Na 1ª vez varre as imagens (≈ 30 s no Mac) e grava o cache `frac_invalido_ir.csv`. Saída em `resultados_janelas/`. |
| `analisar_resultados_kaggle.py` | Analisa o log de um notebook baixado do Kaggle (seção 10.3). Saída em `resultados_kaggle/` (`analise_kaggle.json` e `metricas_por_execucao_log.csv`). **Ver lacuna abaixo.** |
| `cortar_dataset.py` | Corta os 3 `.h5` a 20% **por ciclone** (preserva as sequências) em `data/20percent/`, para testes rápidos (`--frac`, `--seed`, `--compressao`). |
| `treinar_remoto.py` | A parte de **treino** do notebook (seções 1 a 6) como script de linha de comando para uma máquina com GPU (RTX 5090): dados → duplicatas → quadros em cache → janelas → folds → treina fold × seed × variante e grava `runs/*.npz`/`.pt` (mesmos nomes e conteúdo do notebook). Sem Jupyter, matplotlib nem scipy; log também em arquivo; retomável; SIGTERM/Ctrl-C saem limpos (código 130); gravação atômica; confere a GPU antes de ler os dados (Blackwell exige PyTorch cu128). **Não** faz tabelas, estatística nem figuras. Uso na seção 14.8. |
| `verificar_equivalencia_treino.py` | Confere que as definições de `treinar_remoto.py` são idênticas às do notebook (compara as árvores sintáticas; só lê os dois arquivos). Diferenças declaradas dentro dele: `log` (também grava arquivo) e `salvar_execucao` (atômico). Rodar depois de qualquer mudança em um dos dois. |
| `versão final.pdf` | O TCC (ponto de partida e baseline). |
| `data/` | Os 3 arquivos TCIR (≈ 48 GB) + `20percent/`. |

Não versionados (`.gitignore`): `data/`, `resultados_*/`, `__pycache__/`. Em outra máquina é preciso fornecer `data/` (ou usar o dataset do Kaggle); as pastas `resultados_*` se regeneram com os scripts.

**Cronologia:** em 2026-10-02 foram entregues `verificar_duplicatas.py`, `resultados_duplicatas/` e a 1ª versão de `pipeline_tc_kaggle.ipynb`. Em 2026-10-12 vieram a 1ª execução no Kaggle, `estatisticas_janelas.py` (que recriou o código de números antes perdidos), `analisar_resultados_kaggle.py` e os ajustes do notebook (seção 10). Ainda em 2026-10-12: `treinar_remoto.py` e `verificar_equivalencia_treino.py` (treino numa máquina remota com RTX 5090; seção 10.1, item 8).

**Mapa do notebook:** seção 1 configuração (`MODO`, `cfg`) · 2.2 duplicatas (`detectar_duplicatas`) · 2.3 extração IR+PMW · 2.4 janelas e rótulos · 3 tensor (`montar_lote`, `rotacionar_e_recortar`) · 4 modelo (`CodificadorCNN`, LSTM, `CamadaKAN`, `ModeloTC`) · 5 treino de uma execução (5.1 folds, 5.2 perda e métricas, 5.3 `treinar_execucao`) · 6 folds × seeds × variantes · 7 resultados (7.2b baselines sem imagens, 7.3 Wilcoxon + Holm com `holm_bonferroni`, 7.3b bootstrap por ciclone, 7.5 janelas com erro grande) · 8 explicabilidade (8.1 Grad-CAM, 8.2 splines, 8.3 variáveis latentes) · 9 fechamento · 10 empacotar a saída em `.zip`.

**Lacunas conhecidas**
- `treinar_remoto.py` foi testado **só em CPU** (modo teste, `data/20percent/`, macOS): **nunca rodou em GPU CUDA nem na RTX 5090**. Os trechos de CUDA que o notebook não tem (nome/capacidade/VRAM da GPU, aviso de arquitetura sem kernel, pico de memória) usam APIs padrão do PyTorch, mas não foram exercitados. A primeira execução na 5090 deve ser o `--modo teste` (seção 14.8).
- `analisar_resultados_kaggle.py` pertence à raiz do projeto, mas na máquina original a única cópia estava **fora da pasta** (`~/Desktop`). Sem ele, os números da seção 9 não são reproduzíveis a partir do repositório. Restaurar na raiz (ele acha `estatisticas_janelas.py`, `resultados_janelas/` e `data/` relativos à própria pasta). Foi escrito para o log da 1ª execução (75 treinos, 3 variantes, referência `cnn_lstm_kan`, sem remoção de duplicatas): revisar antes de reutilizar.
- O notebook executado da 1ª execução (`kanscnn (3).ipynb`, baixado do Kaggle) e o `resultados_pipeline.zip` dela **não** estão no repositório (o zip nem chegou a ser baixado, por isso não há `runs/*.npz` para investigar a faixa horizontal).
- A variante `cnn_lstm_kan_vmax` (ablação da multitarefa) nunca rodou em escala.

## 5. Ponto de partida: o TCC e o que muda

### 5.1 O TCC (baseline a superar)

- Dataset TCIR (*Tropical Cyclone for Image-to-Intensity Regression*, Chen et al., 2018): 70.501 imagens usadas (2003–2017), 201×201 px, 4 bandas. Splits por ano do ID: treino 2003–2014, validação 2015–2016, teste 2017.
- Pré-processamento: seleção de bandas (IR + PMW), z-score, aumento de dados por rotação e, para injetar informação de movimento em modelos que recebiam **uma imagem isolada por vez**, uma banda de diferença temporal (**DTA ou SDTA**, alternativas excludentes).
- Cinco arquiteturas, com cabeça final de regressão em 3 camadas totalmente conectadas (MLP): CNN-TC, CNN-TC-Δ1 (DTA), CNN-TC-Δ2 (SDTA), ResNet50 e MobileNetV2. Comparação com Wilcoxon pareado (o Shapiro-Wilk rejeitou a normalidade dos erros, Tabela 4 do TCC; a Tabela 5 traz a média dos erros e a % de imagens em que um modelo erra menos). Explicabilidade com SHAP.

| Modelo (teste 2017, N = 4.296) | RMSE (nós) | MAE (nós) | MSE |
|---|---|---|---|
| CNN-TC (baseline) | 10,06 | 7,53 | 101,14 |
| CNN-TC-Δ1 (DTA) | 9,56 | 7,22 | 91,38 |
| CNN-TC-Δ2 (SDTA) | 9,72 | 7,23 | 94,52 |
| ResNet50 (SDTA) | 9,24 | 7,01 | 85,36 |
| **MobileNetV2 (SDTA), melhor** | **9,14** | **7,00** | **83,55** |

O artigo original do CNN-TC (Chen; Chen; Lin, 2018) reporta RMSE 10,58; a reimplementação do TCC obteve 10,06 no mesmo teste de 2017, usada como referência direta.

### 5.2 O que muda em relação ao TCC

A nova pipeline é uma **evolução conceitual** do TCC, não uma reinicialização: mantém o dataset, a seleção de bandas e os canais DTA/SDTA já validados.

| Aspecto | TCC | Nova pipeline |
|---|---|---|
| Entrada | 1 imagem por amostra | sequência de 5 quadros (≈ 3 h de espaçamento) |
| Canais por quadro | 2 (IR+PMW) ou 3 (+ DTA **ou** SDTA) | 4 (IR + PMW + DTA **e** SDTA simultâneos) |
| Modelagem temporal | implícita (só pelo canal de diferença) | explícita (LSTM sobre a sequência de embeddings) |
| Saídas | 1 valor (Vmax) | multitarefa: Vmax e ΔVmax (a especificação original previa também Pmin, fora desta versão) |
| Cabeça de regressão | 3 camadas totalmente conectadas | cabeças KAN independentes |
| Validação | split único por ano (2003–14 / 15–16 / 17) | CV agrupada por ciclone (k = 5) × 5 seeds |
| Teste estatístico | Wilcoxon pareado | Wilcoxon pareado + Holm-Bonferroni (na implementação, o veredito vem do bootstrap por ciclone: seção 8.9) |
| Explicabilidade | SHAP | Grad-CAM (encoder) + splines da KAN |
| Deduplicação de ID | não mencionada | etapa explícita |

## 6. Especificação e conceitos da pipeline

### 6.1 Visão geral

```text
 janela de 5 quadros (t-12h, t-9h, t-6h, t-3h, t), cada quadro = tensor (4, 64, 64)
                         |   (o MESMO encoder para todos: pesos compartilhados)
                         v
                CNN encoder --> e(t-12h) ... e(t)     (um vetor latente por quadro)
                         v
                LSTM:  h_t = LSTM(e(t_i), h_{t-1})  -->  h(t)  (estado final)
                         v
        +----------------+---------------------------+
   KAN Vmax(t)      KAN Pmin(t)               KAN dVmax[t, t+Delta]
   (principal)   (fora desta versão)        (intensificação rápida)
```

Cada um dos 5 quadros passa pelo **mesmo** encoder (não são 5 redes), virando 5 vetores latentes. A LSTM consome esses vetores **em sequência**, carregando memória de um para o seguinte, e produz um único estado final `h(t)` que resume "como o ciclone evoluiu" até agora. Esse estado alimenta cabeças KAN independentes, cada uma prevendo uma grandeza.

### 6.2 Dados e pré-processamento

- **Base:** TCIR, com 4 bandas (IR, WV, VIS, PMW), 201×201 px (≈ 7,8 km/px, ≈ 1.500 km de lado). Rótulos: ID do ciclone, Vmax (nós), tamanho, coordenadas, data/hora e `MSLP` (seção 7.2).
- **Deduplicação por ID.** Uma duplicata é um dado redundante: a rede enxerga o mesmo exemplo como se fossem dois. Isso (1) **vicia o treino** (o gradiente dá peso dobrado àquele exemplo) e (2) **vaza informação na validação cruzada** (se as cópias caírem em folds diferentes, a do treino "entrega a resposta" da do teste e o erro de teste fica artificialmente baixo). Caso esperado: o satélite registrar duas fotos do mesmo ciclone no mesmo instante, gerando duas linhas com o mesmo ID e o mesmo horário. Procedimento: agrupar por ID do ciclone, verificar se há mais de uma imagem no mesmo instante e manter só uma. A especificação original restringia isso ao conjunto de treino (só para o ajuste do modelo); a implementação aplica a todos os dados (seção 11, item 2). Também é preciso garantir **3 h** de diferença entre quadros consecutivos do mesmo ciclone, pré-requisito da sequência.
- **Seleção de bandas: IR + PMW.** VIS é descartada (à noite não tem informação: vem toda `NaN`) e WV também (muito parecida com o IR, o que favorece *overfitting*). O IR mostra a temperatura do topo das nuvens (mais frio = convecção mais profunda); o PMW mostra a chuva, que enxerga *através* das nuvens a estrutura da parede do olho.
- **Limpeza e normalização:** `NaN` e valores > 1000 viram 0 (como no TCC); z-score por canal, $x_{norm}=(x-\mu)/\sigma$.
- **Canais de movimento DTA e SDTA, ao mesmo tempo.** Para o pixel $i$ no instante $n$, sobre o IR:

$$\mathrm{DTA}^n_i=\left|I^n_i-I^{n-1}_i\right|\qquad
\mathrm{SDTA}^n_i=\left|(I^n_i-I^{n-1}_i)-(I^{n-1}_i-I^{n-2}_i)\right|=\left|I^n_i-2I^{n-1}_i+I^{n-2}_i\right|$$

  O DTA (1ª ordem) é a "velocidade" de mudança da nuvem entre dois quadros; o SDTA (2ª ordem) é a "aceleração". A ideia é dar à rede, em cada quadro, não só o "estado" do ciclone (IR, PMW) mas também sua velocidade e aceleração de mudança, complementar à informação temporal que a LSTM extrai *entre* quadros. Usa-se o IR por ter a maior resolução espacial (evita reamostrar o PMW).
- **Rotação aleatória** ($\theta\sim\mathcal U(0°,360°)$): ciclones não têm uma rotação canônica (um furacão girado 90° continua sendo um furacão), então a rede deve aprender **invariância angular**. **Só no treino**, com probabilidade **0,55** (valor a confirmar; a alternativa é 0,5), aplicada ao **tensor inteiro** — não é um canal.
- **Tensor por quadro `(4, 64, 64)`:** canal 1 = IR, 2 = PMW, 3 = DTA, 4 = SDTA. Recorte 64×64 centrado no olho (≈ 2° de lat/lon de raio, ≈ 28 px; arredondado para potência de 2, mesmo recorte do TCC): é a região do olho e da parede do olho, que é o que mais pesa na intensidade (como o SHAP do TCC mostrou).

### 6.3 Sequência e CNN encoder

- **Janela deslizante** de **5 quadros** com Δt ≈ 3 h: t−12h, t−9h, t−6h, t−3h, t. "Quadro" (*frame*) é cada imagem da sequência: o ciclone é tratado como uma sequência de quadros evoluindo no tempo.
- Cada quadro passa pelo **mesmo** CNN encoder (mesmos pesos aplicados 5 vezes), que o "amassa" num **vetor latente** (*embedding*) $e(t_i)$. Analogia: usar os mesmos óculos para olhar 5 fotos; aprende-se *uma* forma de descrever um quadro de ciclone, e ligar as 5 descrições numa "história" é papel da LSTM.
- O encoder pode reaproveitar a arquitetura de uma das CNNs do TCC (CNN-TC, ResNet50 ou MobileNetV2), sem a camada final de regressão. Como roda 5× por amostra, uma arquitetura leve (estilo CNN-TC, entrada 64×64) é bem mais viável que ResNet50/MobileNetV2 a 224×224 repetida 5×.

### 6.4 Dinâmica temporal: LSTM

- **Por que não tratar cada quadro isoladamente?** A intensidade é consequência de um *processo dinâmico*: importa como as nuvens mudaram, não só como estão. É preciso um mecanismo que "lembre" o que veio antes ao processar o que vem depois.
- **RNN simples:** $h_t=\tanh(W_x x_t+W_h h_{t-1}+b)$. No treino (*backpropagation through time*), o gradiente volta por muitos passos sendo multiplicado repetidamente pela mesma matriz $W_h$, e por isso **some** (a rede "esquece" o passado) ou **explode**.
- **LSTM** (Hochreiter & Schmidhuber, 1997; sucessora da RNN) resolve isso com um **estado de célula** $c_t$ (uma "esteira" que carrega informação com poucas interferências) controlado por três **portões**, cada um com saída sigmoide entre 0 e 1 (uma válvula: 0 bloqueia, 1 deixa passar):

$$\begin{aligned}
f_t&=\sigma(W_f[h_{t-1},x_t]+b_f) &&\text{portão de esquecimento: quanto da memória antiga manter}\\
i_t&=\sigma(W_i[h_{t-1},x_t]+b_i),\quad \tilde c_t=\tanh(W_c[h_{t-1},x_t]+b_c) &&\text{portão de entrada: quanta informação nova gravar}\\
c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t &&\text{atualização da esteira (memória de longo prazo)}\\
o_t&=\sigma(W_o[h_{t-1},x_t]+b_o),\quad h_t=o_t\odot\tanh(c_t) &&\text{portão de saída: o que expor como estado}
\end{aligned}$$

  Como $c_t$ só sofre somas e multiplicações por portões, o gradiente atravessa muitos passos sem sumir.
- **No projeto:** $x_t$ é o vetor $e(t_i)$ do encoder (não pixels). Notação do diagrama: $h_t=\mathrm{LSTM}(e(t_i),h_{t-1})$. A LSTM lê `t-12h → t-9h → t-6h → t-3h → t` em ordem; o estado final $h(t)$ resume a evolução e é a **única** entrada das cabeças. Como $h_t=o_t\odot\tanh(c_t)$, fica sempre em $(-1,1)$: a faixa que a 1ª camada da KAN usa como domínio.
- **Limitação assumida: horizonte fixo e discreto.** A LSTM supõe passos regulares (aqui, 3 h) e não sabe "quanto tempo passou": um *gap* de 6 h é tratado como se fosse de 3 h e confunde a dinâmica aprendida. A concepção original usava tempo contínuo (equações diferenciais, que lidam bem com irregularidade); foi simplificada para LSTM por viabilidade, e essa é a **única modificação** em relação à concepção original, aceita por ora. A pipeline só monta janelas sem buracos.
- **Trabalho futuro: Neural ODE** (Chen et al., 2018): o estado oculto é a solução de $dh/dt=f(h,t)$, integrável sobre qualquer intervalo, lidando com amostragem irregular. Citar nas limitações e nos trabalhos futuros do artigo.

### 6.5 Cabeças KAN multitarefa

**Aprendizado multitarefa.** Um único **tronco** compartilhado (encoder + LSTM) e uma **cabeça** especializada por tarefa, todas lendo o mesmo `h(t)`. Vantagens: (1) a representação compartilhada é forçada a capturar o que serve a **todas** as tarefas, o que funciona como regularizador; (2) vento, pressão e intensificação rápida são fisicamente correlacionados (relações vento-pressão bem estabelecidas); (3) um modelo em vez de vários.

| Cabeça | Alvo | Natureza | Situação |
|---|---|---|---|
| 1 | $V_{max}(t)$, vento máximo no instante atual | *nowcasting* (mesma tarefa do TCC) | principal e obrigatória |
| 2 | $P_{min}(t)$, pressão central mínima no instante atual | *nowcasting*, tarefa auxiliar nova | **fora desta versão** (decisão do projeto); a coluna `MSLP` do TCIR já fornece o rótulo |
| 3 | $\Delta V_{max}[t,t+\Delta]$, variação futura | *forecasting*: intensificação rápida | $\Delta=24$ h (a confirmar); só `h(t)` como entrada, o valor futuro é apenas rótulo |

As cabeças além da principal são negociáveis: podem ser mantidas, removidas ou trocadas. **Intensificação rápida (RI):** operacionalmente, aumento de Vmax ≥ 30 nós em 24 h (Kaplan & DeMaria, 2003). É um dos fenômenos mais difíceis de prever e mais perigosos (surpreende alertas e populações: furacões Patricia 2015, Otis 2023); prever a **taxa de mudança futura**, e não só o estado atual, é um ponto forte para a narrativa do artigo.

**O que é uma KAN.** Numa camada de MLP cada neurônio soma as entradas com **pesos escalares** e aplica uma ativação **fixa**; a não linearidade mora no **nó**: $x_{l+1,j}=\sigma\big(\sum_i w_{l,j,i}x_{l,i}+b_{l,j}\big)$. Numa **KAN** (*Kolmogorov-Arnold Network*, Liu et al., 2024) inverte-se a lógica: cada **aresta** carrega uma função univariada **aprendível** $\varphi_{l,j,i}$ (tipicamente uma *spline* com pontos de controle treináveis) e o nó só soma: $x_{l+1,j}=\sum_i\varphi_{l,j,i}(x_{l,i})$. A inspiração é o teorema de representação de Kolmogorov-Arnold (1957): toda função contínua multivariada pode ser escrita como somas e composições de funções contínuas de **uma** variável, $f(x_1,\dots,x_n)=\sum_{q=0}^{2n}\Phi_q\big(\sum_{p=1}^n\varphi_{q,p}(x_p)\big)$. A KAN generaliza essa estrutura de 2 camadas para uma pilha de camadas, treinada por *backpropagation* como uma rede comum.

**Por que KAN aqui.** (1) **Interpretabilidade nativa:** cada $\varphi$ é uma função de uma variável que se pode desenhar, mostrando que relação a rede aprendeu entre cada variável latente e cada saída, sem método *post-hoc* separado. (2) **Novidade metodológica:** KAN é de 2024 e seu uso em regressão multitarefa de ciclones é novo, o que a torna o cerne da proposta. (3) **Custo:** cada aresta guarda uma spline com vários coeficientes em vez de 1 escalar, então KANs tendem a treinar mais devagar que MLPs de tamanho comparável; aqui isso é desprezível, porque a KAN só vê 128 números por amostra (o caro é a CNN).

### 6.6 Validação e estatística

- **CV agrupada por ciclone (k = 5).** Num k-fold comum, janelas vizinhas (mesma tempestade, janelas sobrepostas, aparência parecida) cairiam em lados opostos e o modelo seria "testado" em imagens que já viu: **vazamento** que infla o desempenho. Agrupa-se por ID do ciclone (`GroupKFold`): todas as janelas de um ciclone ficam inteiras no mesmo lado em cada fold. Testa-se, com honestidade, a generalização para ciclones **nunca vistos**, o cenário real de uso.
- **5 seeds por fold.** Treinar redes é estocástico (pesos iniciais, ordem dos lotes e rotações aleatórios; em GPU, somas de ponto flutuante em ordem não fixa): **uma seed fixa não garante o mesmo resultado**. Repetir com 5 seeds mede quanto da diferença entre modelos é real e quanto é ruído do treino. Total: 5 folds × 5 seeds = 25 treinos por variante.
- **Wilcoxon pareado** (o mesmo do TCC): alternativa não paramétrica ao *t* pareado, adequada porque os erros absolutos não são normais e os dados são pareados (mesmo exemplo de teste, dois modelos). Ranqueia as diferenças e testa se a distribuição é simétrica em torno de zero (sem diferença sistemática) ou enviesada.
- **Correção de Holm-Bonferroni.** Muitas comparações simultâneas inflam o falso positivo (20 testes independentes com α = 5% dão ≈ 64% de chance de pelo menos um). Procedimento: ordenar os p-valores $p_{(1)}\le\dots\le p_{(m)}$; comparar $p_{(k)}$ com $\alpha/(m-k+1)$; **parar** no primeiro que não passa (rejeita só os anteriores). Menos conservadora que a Bonferroni simples e igualmente rigorosa quanto ao erro familiar. Em p-valores ajustados: $p^{adj}_{(k)}=\max_{j\le k}\min\{1,(m-j+1)p_{(j)}\}$, significativo se $p^{adj}\le\alpha$. Em Python: `statsmodels.stats.multitest.multipletests(method='holm')`.

### 6.7 Explicabilidade

- **Grad-CAM no encoder** (Selvaraju et al., 2017): gradiente da saída em relação aos mapas de características de uma camada convolucional (a última antes do achatamento) → média espacial dá um **peso por canal** → soma ponderada dos mapas + ReLU → mapa de calor de baixa resolução, ampliado ao tamanho da entrada, que mostra **onde** a rede olhou. Cumpre o papel do SHAP do TCC com custo muito menor (gradiente × teoria dos jogos), embora menos rigoroso (heurístico, não axiomático).
- **Funções spline da KAN:** as $\varphi$ aprendidas se desenham diretamente e mostram **que relação** o modelo aprendeu entre cada variável latente de `h(t)` e cada saída. Grad-CAM responde "onde"; as splines respondem "que relação": são complementares.

| | TCC (SHAP) | Esta pipeline |
|---|---|---|
| Método | valores de Shapley (teoria dos jogos) | gradientes (Grad-CAM) + splines nativas (KAN) |
| O que explica | contribuição de cada pixel para a saída única | onde a CNN olhou, por instante + que função a KAN aprendeu |
| Custo | alto | baixo (um passe de gradiente) + gratuito (as splines já fazem parte do modelo) |
| Rigor teórico | axiomático | heurístico, mas barato |
| Aplicado a | melhor modelo (MobileNetV2) | encoder da pipeline e cabeças KAN |

### 6.8 Cuidados de implementação e como o código os garante

| # | Cuidado | Por quê | Como o código garante |
|---|---|---|---|
| 1 | Mesma rotação em todos os canais do quadro | rotações independentes por canal destruiriam a correspondência entre IR, PMW, DTA e SDTA | os 4 canais são empilhados antes de girar |
| 2 | Mesma rotação em todos os 5 quadros da janela | senão a sequência fica espacialmente incoerente e a LSTM não aprende a dinâmica | 5 quadros × 4 canais = 20 "canais" de **uma** imagem; uma só chamada de `grid_sample` com um ângulo por janela |
| 3 | Diferença (DTA/SDTA) **antes** da rotação | girar cada quadro do par separadamente e só depois subtrair mediria o artefato da rotação, não o movimento do ciclone | `montar_lote` calcula DTA/SDTA na grade original 92×92 e só depois gira e recorta (mesma ordem da Figura 3 do TCC) |
| 4 | Deduplicação **antes** da janela deslizante | um quadro duplicado numa janela quebraria o espaçamento regular de 3 h que a LSTM pressupõe | a detecção roda na construção da tabela, antes de `construir_janelas`; um `assert` recusa (ciclone, instante) repetido |
| 5 | Sem vazamento na cabeça de ΔVmax | o encoder/LSTM nunca deve receber quadros futuros; "h(t)→Δ" do diagrama é o horizonte do **alvo** (a confirmar) | as cabeças só recebem `h(t)`; `Vmax(t+24h)` vem da tabela, só como rótulo |
| 6 | Framework | o TCC era Keras/TensorFlow; as KANs de referência (`pykan`, `efficient-kan`) são PyTorch | PyTorch |

### 6.9 Ordem seguida na implementação

(1) Camada de dados: deduplicação e tensor de canais. (2) Pipeline principal: janela deslizante → encoder → LSTM → cabeças KAN (podendo começar só com Vmax). (3) Por último: CV × seeds, testes estatísticos e explicabilidade. As três etapas estão implementadas; falta rodar em escala e analisar.

## 7. Dados (TCIR) e achados

### 7.1 Formato

Três arquivos HDF5, cada um com `matrix` `(N, 201, 201, 4)` float32 (canais 0 = IR em K, ≈ 184–297 K numa imagem típica; 1 = WV; 2 = VIS, toda `NaN` à noite; 3 = PMW em mm/h) e `info`, uma linha por imagem: `data_set` (a **bacia**, não treino/teste), `ID` (p. ex. `201616L` = 2016, ciclone 16, bacia L), `lon`, `lat`, `time` (`AAAAMMDDHH`), `Vmax`, `R35_4qAVG` (raio de ventos de 35 nós) e `MSLP`.

| Arquivo | Imagens | IDs | Anos | Bacias |
|---|---|---|---|---|
| `TCIR-ATLN_EPAC_WPAC-001.h5` | 47.381 | 861 | 2003–2016 | Atlântico, Pacífico Leste e Oeste |
| `TCIR-CPAC_IO_SH-002.h5` | 23.118 | 424 | 2003–2016 | Pacífico Central, Índico, Hemisfério Sul |
| `TCIR-ALL_2017-003.h5` | 4.580 | 94 | 2017 | todas |
| **Total** | **75.079** | **1.379** | | |

### 7.2 Fatos verificados nos dados reais

- **Espaçamento:** dos 73.700 intervalos entre imagens do mesmo ciclone, 73.698 são de exatamente 3 h (só duas lacunas: 9 h e 15 h). A preocupação com intervalos irregulares praticamente não se aplica ao TCIR em si; o que cria buracos são os quadros inválidos (seção 7.4).
- **`MSLP` existe e está completa** (872–1024 hPa, nenhum ausente): o rótulo de Pmin já está no TCIR, sem precisar de fonte externa como o IBTrACS. A cabeça de Pmin ficou de fora por decisão do projeto, mas poderia ser acrescentada sem custo de dados.
- O `info` é gravado no formato "fixed" do pandas, com as colunas de texto dentro de um *pickle*. `pandas.read_hdf` falha com pandas 3.x + PyTables (o atributo `pandas_type` vem como `bytes`): tanto os scripts quanto o notebook leem o `info` direto com `h5py`, com um *unpickler* restrito que só aceita os 3 objetos NumPy necessários (desserializar pickle arbitrário é inseguro).
- **Splits do TCC** (por ano do ID): treino 2003–2014 (1.097 ciclones), validação 2015–2016 (188), teste 2017 (94). O TCC usou 70.501 imagens e teste com N = 4.296 (o TCIR tem 4.580 imagens de 2017): isso sugere que o TCC aplicou algum filtro, ainda não conferido. Neste projeto o split fixo é substituído pela CV agrupada por ciclone; os splits por ano só servem de referência.
- Cerca de 80% dos Vmax são múltiplos de 5 nós (a incerteza do *best track* limita o RMSE alcançável).

### 7.3 Verificação de duplicatas

| | Verificação | Custo |
|---|---|---|
| **A** | **Duplicata estrita:** o par `(ID, instante)` aparece mais de uma vez | instantâneo |
| **B** | **Mesmo ciclone físico com IDs diferentes:** IDs distintos, mesmo instante, posição a ≤ 0,25° (≈ 28 km) | instantâneo |
| **C** | **Comparação de pixels** dos candidatos de A e B: idênticos bit a bit? correlação do IR? | segundos |
| **D** | **Varredura por hash** (`--varrer-imagens`, só no script): hash de 128 bits de cada uma das 75.079 imagens, agrupando as repetidas; pega duplicatas exatas mesmo que os metadados não denunciem | ≈ 100 s (lê 48 GB) |

O notebook faz A e B (só a tabela `info`) e C de forma informativa (lê as poucas imagens removidas).

**Por que B:** um ciclone que cruza de uma bacia para outra (p. ex. do Atlântico para o Pacífico Leste na América Central) pode receber **um ID em cada bacia**; na transição os dois catálogos descrevem a mesma tempestade. Para a rede são duas amostras iguais; para a CV agrupada por ID, são dois "ciclones" que podem cair em folds diferentes. **Por que 0,25°:** um pixel do TCIR tem ≈ 0,07°, então 0,25° ≈ 3,5 pixels; nos dados há folga nítida (os 72 pares duplicados ficam a no máximo 0,100°; o par de ciclones *diferentes* mais próximo fora da tolerância está a 0,93°).

**Como decide o que manter:** os IDs que compartilham ao menos um par candidato são unidos (*union-find*) num mesmo "ciclone físico"; o `id_canonico` é o ID com mais registros (empate: o menor ID). Dentro do ciclone, linhas no mesmo instante são duplicatas; fica uma por instante (a do ID com mais registros) e as outras recebem `acao = remover`.

**Resultados:**
- **A: 0 duplicatas** (nenhum `(ID, instante)` se repete, nem dentro de um arquivo nem entre arquivos). A definição estrita não se confirma nos dados.
- **B: 2 ciclones físicos, 72 registros redundantes:**

| IDs | Bacias | Linhas por ID | Instantes duplicados | Período | Split do TCC | Imagens idênticas |
|---|---|---|---|---|---|---|
| `201010L` + `201011E` | ATLN + EPAC | 43 / 9 | 1 | 2010-09-04 18h | treino | 1/1 |
| `201616L` + `201622E` | ATLN + EPAC | 71 / 71 | 71 | 2016-11-17 18h → 2016-11-26 12h | validação | 69/71 |

  O segundo é o ciclone inteiro duplicado (as datas coincidem com o furacão Otto de 2016, que cruzou a América Central). Registros a remover por split do TCC: treino 1, validação 71, teste 0.
- **C:** dos 72 pares, **70 são idênticos bit a bit** nos 4 canais; 2 são quase-duplicatas (mesma hora, posição a ≈ 0,1°, recorte deslocado 1–2 px, correlação do IR 0,95 e 0,94), também tratadas como redundantes.
- **D:** 72 grupos de imagens completas idênticas (537 linhas): 70 são os pares acima e 2 são imagens vazias repetidas (seção 7.4). **Zero grupos não explicados:** não existe outra duplicata exata no dataset.

**Como a pipeline usa o resultado** (notebook, seção 2.2, função `detectar_duplicatas`; resultado idêntico ao do script):
1. **Remove** as linhas repetidas (72 registros): fica a do ID com mais registros.
2. **Une os IDs:** `201622E → 201616L` e `201011E → 201010L`. A série temporal passa a ser a união das linhas dos IDs equivalentes.
3. Usa o `id_ciclone` unificado como **grupo da CV**: as duas cópias nunca ficam em folds diferentes.

A remoção vale para **todos** os dados, e não só para o treino como previa a especificação original: a janela deslizante exige instantes únicos e regulares (um instante duplicado no teste quebraria a regularidade de 3 h tanto quanto no treino); afeta 72 de 75.079 registros (0,1%). Com `cfg.detectar_duplicatas = False` o notebook **avisa em destaque** e segue sem remover nada; se restar `(ciclone, instante)` repetido, um `assert` interrompe a execução. Ao fim, o log registra quantos removidos são idênticos bit a bit à cópia mantida (**70 de 72**). Tolerância de posição: `cfg.tol_duplicata_graus` (0,25°). Sem tratamento, o furacão Otto entraria **duas vezes** nos dados e uma cópia poderia cair no treino e a outra no teste.

### 7.4 Imagens vazias (achado extra)

**479 imagens (0,64%)**, em **63 ciclones**, têm o canal IR **100% vazio** no recorte central (treino 441, validação 20, teste 18); 397 delas são (quase) integralmente `NaN` nos 4 canais e idênticas entre si (provavelmente instantes sem cobertura do satélite). Com a limpeza do TCC (`NaN → 0`) virariam um quadro todo zerado que ainda carrega rótulo de Vmax: o modelo seria treinado a "prever a velocidade de uma imagem em branco". Além disso, ≈ 4,5% das imagens têm uma fração pequena de IR ausente (linhas de varredura do satélite).

**Regra:** **quadro com mais de 10% de pixels de IR ausentes é inválido** (`cfg.max_frac_nan_ir`) e sai da série. Isso abre um buraco de ≥ 6 h ao lado dele, e nenhuma janela atravessa buracos. São **549 quadros inválidos** (0,73%), que custam **1.085 janelas** (de 66.755 possíveis se todos os quadros fossem válidos).

### 7.5 Números do dataset após as regras da pipeline

| Item | Valor |
|---|---|
| Imagens no TCIR | 75.079 (1.379 IDs) |
| Após remover duplicatas e unir IDs | 75.007 imagens, 1.377 ciclones |
| Quadros inválidos (IR com > 10% ausente) | 549 |
| **Janelas utilizáveis (5 quadros; 7 lidos)** | **65.670**, em 1.373 ciclones (mediana 45 por ciclone; máx. 195) |
| Janelas com rótulo de ΔVmax | 55.072 |
| RI (ΔVmax ≥ 30 nós em 24 h) | 3.669 = **6,66%** das rotuladas |
| Vmax(t) das janelas | média **51,77**, dp **28,90** nós (mín 10; Q25 30; mediana 43; Q75 65; máx 185) |
| ΔVmax = Vmax(t+24h) − Vmax(t) | média −0,39, dp 19,72 (55.072 janelas) |
| Vmax por split (média ± dp) | treino 51,0 ± 28,7 · validação 56,3 ± 30,2 · teste 50,8 ± 27,1 |
| Janelas por split do TCC | treino 52.275 · validação 9.428 · teste 3.967 |
| CV (5 folds) | ≈ 13.134 janelas e 274–276 ciclones por fold |

**Como é calculada a média/dp do Vmax:** cada janela tem **um** rótulo, o Vmax (nós, do *best track* do TCIR) do **último** quadro, `t`; juntam-se os rótulos de **todas** as janelas utilizáveis (a unidade é a janela, não o ciclone, então ciclones longos pesam mais); média = soma ÷ 65.670 e dp com divisor N − 1 (com N tão grande, o populacional dá o mesmo, 28,90).

**Como ler:** 33% das janelas são depressões (< 34 nós), 40% tempestades tropicais (34–63), 11% categoria 1 e só 16% categoria 2 ou mais. O dp de 28,9 nós é a **régua mínima de erro**: um "modelo" que sempre prevê a média teria RMSE ≈ 28,9, contra ≈ 10 do TCC. A normalização do treino usa média/dp **só do treino de cada fold**, não estes números.

## 8. Como está implementado

### 8.1 Pré-processamento no notebook

Parte acontece **uma vez** (extração e cache), parte **a cada lote** (normalização, diferenças e rotação, na GPU).

- **Extração e limpeza (com cache).** Para cada imagem lê só as **linhas** do recorte central 92×92 (trecho contíguo no disco; ≈ 2.700 imagens/s no teste local) e corta as colunas na memória. Pixels `NaN` ou > 1000 viram inválidos (guardados como `NaN` em `float16`). Resultado: `(75.079, 2, 92, 92)` em `float16`, ≈ 2,5 GB, num cache em disco; guarda-se também a **fração de IR inválido** de cada quadro (usada na regra de quadro inválido).
- **Pixels inválidos dentro de um quadro válido** (`cfg.tratamento_invalidos`): `"tcc"` (padrão) faz `NaN → 0` **antes** do z-score, exatamente como o TCC (efeito colateral: o IR vira 0 K, ≈ −7,7σ após o z-score, e DTA/SDTA ali ficam enormes, aparecendo como faixas retas brilhantes, as linhas de varredura do satélite; a rede aprende a conviver, como no TCC); `"mascara"` faz `NaN →` média (z = 0) e DTA/SDTA = 0 nesses pixels (alternativa limpa para ablação).
- **z-score global por canal**, com μ e σ calculados numa amostra de ≈ 3.000 quadros do **treino de cada fold** (nada do teste entra). Global, e não por imagem: normalizar cada imagem por sua média apagaria justamente a **temperatura absoluta** do topo das nuvens. Valores típicos: IR μ ≈ 252 K, σ ≈ 32; PMW μ ≈ 1,2 mm/h, σ ≈ 2,5.
- **DTA/SDTA** sobre o IR já normalizado, na grade original (seção 6.2).
- **Rotação:** `affine_grid` + `grid_sample` (interpolação bilinear), ângulo $\theta\sim\mathcal U(0°,360°)$, só no treino, com probabilidade `cfg.prob_rotacao = 0,55`. Com ângulo 0 (ou 90°) cada pixel de saída cai exatamente num pixel de entrada, então o recorte é exato.
- **Ordem das operações e por que importa:**

```text
 (92x92, grade original)                          (gira)      (corta)
 IR, PMW -> z-score -> DTA, SDTA -> empilha 5x4 canais -> rotação -> recorte 64x64
```

  1. **Diferença antes da rotação** (DTA/SDTA comparam o *mesmo pixel* em instantes diferentes: as imagens precisam estar alinhadas). 2. **Uma rotação para tudo:** 5 quadros × 4 canais = 20 canais empilhados numa só imagem e girados de uma vez, com **um** ângulo por janela. 3. **Rotação antes do corte** (como no TCC): por isso se guarda 92×92 e não 64×64 — o círculo circunscrito a um quadrado 64×64 tem raio $64\sqrt2/2\approx45{,}3$ px, menor que 46 (metade de 92), então qualquer ângulo produz um 64×64 **sem cantos pretos**.
- **Tensor final por janela:** `(5, 4, 64, 64)` = 5 instantes × `[IR, PMW, DTA, SDTA]` × 64×64 px. O notebook gera duas figuras de conferência (`01_tensor_normal.png`, `01_tensor_giro.png`; na 2ª todos os canais e quadros giram **juntos**).
- **Tudo na GPU:** o array de quadros inteiro (2,5 GB em `float16`) fica na memória da GPU e cada lote é montado por indexação: sem `DataLoader`, sem processos auxiliares, sem a CPU como gargalo.

### 8.2 Janelas e rótulos

- **Janela deslizante:** a janela avança de 3 em 3 h pela vida do ciclone; um ciclone com 60 imagens gera ≈ 54 janelas, quase todas sobrepostas (4 dos 5 quadros em comum com a vizinha).
- **Cada janela exige 7 quadros consecutivos** (`t−18h … t`): cada quadro da janela tem 4 canais, inclusive o mais antigo (`t−12h`), cujo SDTA precisa de dois quadros anteriores (`t−15h`, `t−18h`); os 2 extras só participam do cálculo de DTA/SDTA.
- **Continuidade:** uma **corrida** é um trecho de um mesmo ciclone com quadros válidos exatamente a cada 3 h; a janela só existe se os 7 quadros estiverem na mesma corrida. Em vez de tentar consertar buracos, descartam-se essas janelas.
- **Rótulos:** `vmax` = $V_{max}(t)$ em nós (tarefa principal); `dvmax` = $V_{max}(t+24h)-V_{max}(t)$. O rótulo futuro é buscado na série **completa** do ciclone (inclusive em quadros de imagem inválida: o rótulo do *best track* continua valendo); se o ciclone acabou antes de `t+24h`, é `NaN`. **Máscara na perda:** janelas sem ΔVmax seguem treinando `vmax`; a perda de `dvmax` as ignora. O valor Δ = 24 h é `cfg.horizonte_horas`. Os rótulos futuros **nunca** são entrada da rede.

### 8.3 Arquitetura (`ModeloTC`, ≈ 1,08 M parâmetros)

Encoder 913.856 + LSTM 132.096 + 2 cabeças KAN × 18.576.

**CNN encoder** (estilo CNN-TC do TCC: convoluções com *stride* 2 e sem *pooling*, que joga fora informação em imagens de ciclone, que não têm bordas nítidas). Os 5 quadros de cada janela são empilhados no eixo do lote (`x.flatten(0, 1)`: `(B, 5, 4, 64, 64)` → `(B·5, 4, 64, 64)`), a mesma CNN processa tudo e `.view(B, 5, 128)` devolve a sequência.

| Camada | Saída | Parâmetros |
|---|---|---|
| entrada | (4, 64, 64) | |
| Conv 3×3, *stride* 2, 32 + BatchNorm + ReLU | (32, 32, 32) | 1.216 |
| Conv 3×3, *stride* 2, 64 + BatchNorm + ReLU | (64, 16, 16) | 18.560 |
| Conv 3×3, *stride* 2, 128 + BatchNorm + ReLU | (128, 8, 8) | 73.984 |
| Conv 3×3, *stride* 2, 256 + BatchNorm + ReLU | (256, 4, 4) | 295.424 |
| Flatten → Dropout(0,1) → Linear(4096→128) → **LayerNorm** | (128) | 524.672 |

Diferenças deliberadas em relação ao CNN-TC do TCC: sem a camada final de regressão (devolve o vetor, não a velocidade); BatchNorm e Dropout (treino mais estável, menos *overfitting*); **LayerNorm no embedding**: sem ela os embeddings chegavam a ≈ 11, os portões da LSTM saturavam e as variáveis de `h(t)` ficavam coladas em ±1 (o RMSE de validação oscilava 17,8 → 21,2 nas épocas 2–3, contra 15,4 → 14,0 com a LayerNorm) e as splines da KAN ficavam sem informação (só havia dados nas bordas do intervalo); com a LayerNorm, `h(t)` ocupa todo o (−1, 1). Trocar o encoder (ResNet50/MobileNetV2) é trocar a classe `CodificadorCNN`; fica 5× mais caro por amostra.

**LSTM:** 128 unidades, lê `e(t−12h) … e(t)` em ordem; estado final `h(t)` (128 números) em (−1, 1).

**Cabeça KAN `[128 → 16 → 1]`** (implementação própria, `CamadaKAN`): cada aresta é $\varphi(x)=w_b\,\mathrm{silu}(x)+\sum_{m=1}^{G+k}c_m B_m(x)$, com $B_m$ **B-splines** de grau $k=3$ sobre um grid de $G=5$ intervalos (8 funções-base por aresta), $c_m$ os **coeficientes aprendidos** (a forma da curva) e $w_b\,\mathrm{silu}(x)$ um "caminho base" suave que ajuda o treino e dá o comportamento fora do grid. Recursão de Cox-de Boor, com nós igualmente espaçados:

$$B_{i,0}(x)=\begin{cases}1&t_i\le x<t_{i+1}\\0&\text{caso contrário}\end{cases}\qquad
B_{i,k}(x)=\frac{x-t_i}{t_{i+k}-t_i}B_{i,k-1}(x)+\frac{t_{i+k+1}-x}{t_{i+k+1}-t_{i+1}}B_{i+1,k-1}(x)$$

Dentro do grid as bases somam 1 (partição da unidade): cada ponto é uma média ponderada de ≈ 4 coeficientes vizinhos. A implementação foi **conferida contra `scipy.interpolate.BSpline.design_matrix`** (diferença máx. ≈ 1e-7). A 1ª camada ($128\times16$ arestas) tem grid em $(-1,1)$, a faixa de $h(t)$; a 2ª ($16\times1$) recebe somas não limitadas e usa grid em $(-3,3)$. Cada cabeça tem $128\cdot16\cdot(1+8)+16\cdot(1+8)=18.576$ parâmetros e roda sempre em `float32` (a recursão das splines é sensível a `float16`). **Por que escrever a KAN em vez de usar `pykan`/`efficient-kan`:** rodar no Kaggle sem internet nem instalação e manter o código legível e comentado (≈ 40 linhas).

### 8.4 Multitarefa e perda

Alvos padronizados (média 0, dp 1, calculados no treino):

$$\mathcal L=w_V\;\mathrm{MSE}\big(\hat V_{max},V_{max}\big)+w_\Delta\;\mathrm{MSE}\big(\hat{\Delta V}_{max},\Delta V_{max}\big)_{\text{só nas janelas com rótulo em }t+24h}$$

com $w_V=1{,}0$ e $w_\Delta=0{,}5$ (`cfg.pesos_tarefas`): a tarefa principal pesa mais; os gradientes das duas perdas se somam no tronco. Se a intenção for "sem multitarefa nenhuma" (e não só "sem Pmin"), a variante `cnn_lstm_kan_vmax` treina a mesma arquitetura só com Vmax.

### 8.5 Variantes (ablações)

Os testes estatísticos pedem comparação entre modelos; as 4 variantes isolam, uma a uma, cada decisão da proposta (mesmo tronco CNN, mesmos dados). Padrão do notebook: `cfg.variantes = ("cnn_lstm_kan", "cnn_lstm_mlp", "cnn_kan_1frame", "cnn_lstm_kan_vmax")`; referência: `cnn_lstm_kan`.

| Variante | Temporal | Cabeça | Tarefas | Isola o efeito de | Parâmetros |
|---|---|---|---|---|---|
| `cnn_lstm_kan` | LSTM | KAN | Vmax + ΔVmax | **a proposta completa** | 1.083.104 |
| `cnn_lstm_mlp` | LSTM | MLP `[128→64→1]` | Vmax + ΔVmax | a **KAN** | 1.062.594 |
| `cnn_kan_1frame` | nenhum (só o quadro `t`) | KAN | Vmax + ΔVmax | a **sequência/LSTM** | 967.520 |
| `cnn_lstm_kan_vmax` | LSTM | KAN | só Vmax | a **multitarefa** | 1.064.528 |

A `cnn_kan_1frame` ainda recebe DTA/SDTA (movimento via diferenças), então é um *baseline* forte "estilo TCC com 4 canais"; a diferença para a proposta mede o ganho de modelar explicitamente a sequência. `variantes` não entra na assinatura das execuções: dá para acrescentar ou tirar variantes a qualquer momento (as já feitas são reaproveitadas). Além das variantes, o notebook compara com **baselines sem imagens** para ΔVmax (seção 10.1).

### 8.6 Treino de uma execução (`treinar_execucao(variante, fold, seed, …)`)

1. **Semente** `fixar_semente(100·seed + fold)`: afeta inicialização, ordem dos lotes e sorteio das rotações; a divisão em folds **não** depende da seed.
2. **Divisão** treino / validação interna / teste (seção 8.8).
3. **Estatísticas só do treino:** μ e σ de IR e PMW e média/dp dos alvos.
4. **Modelo, otimizador e agenda:** AdamW (lr de pico 1e-3, *weight decay* 1e-4); agenda *OneCycle* (aquece nos primeiros 15% dos passos e decai suavemente até quase zero); **corte de gradiente** em norma 1,0 (importante em redes recorrentes).
5. **Laço de épocas:** embaralha as janelas de treino; por lote de 128, `montar_lote(treino=True)` (com rotação), *forward*, perda multitarefa, *backward*, passo; o último lote incompleto é descartado (BatchNorm).
6. **Precisão mista (GPU CUDA):** CNN e LSTM em `float16` com `GradScaler`; **cabeças em `float32`**.
7. **Early stopping:** ao fim de cada época mede o RMSE de Vmax na validação interna, guarda os melhores pesos e para após `paciencia` (8) épocas sem melhora; o critério é Vmax para todas as variantes (seleção comparável). Com *OneCycle*, a melhor época costuma ficar perto do fim.
8. **Avaliação:** recarrega os melhores pesos e prevê as janelas do fold de teste (sem rotação), voltando os alvos para nós; prevê também a **validação interna** (grava `y_dvmax_val` e `pred_dvmax_val` no `.npz`), com a qual se escolhe o limiar de RI sem consultar o teste.

### 8.7 Métricas (em nós)

- **RMSE** $=\sqrt{\tfrac1n\sum(y-\hat y)^2}$ (a do TCC e da literatura; penaliza erros grandes), **MAE** $=\tfrac1n\sum|y-\hat y|$ e **viés** $=\tfrac1n\sum(\hat y-y)$.
- **RI:** trata-se "ΔVmax ≥ 30 nós" como evento e mede-se **POD** (acertos ÷ eventos reais), **FAR** (alarmes falsos ÷ alarmes) e **CSI** $=\mathrm{TP}/(\mathrm{TP}+\mathrm{FN}+\mathrm{FP})$, de **três jeitos**: (i) alarme em ΔVmax previsto ≥ 30 (o "ingênuo"); (ii) alarme num **limiar ajustado na validação interna** (varre 5…30 nós e fica com o que maximiza o CSI lá; colunas com sufixo `_aj`; o teste nunca é consultado); (iii) **sem limiar**: **AUC-ROC** (probabilidade de um evento real receber escore maior que um não-evento; 0,5 = sem habilidade) e **precisão média (AP)** (área sob a curva precisão × revocação; sem habilidade ≈ prevalência ≈ 0,067; mais informativa que a AUC para eventos raros, Davis e Goadrich, 2006), usando o próprio ΔVmax previsto como escore.
- Um regressor treinado com erro quadrático tende a **regredir à média** em eventos raros (RI é 6,7% das amostras): a POD fica baixa mesmo quando o RMSE de ΔVmax melhora. O limiar ajustado e a AUC/AP medem o que a rede *ordena* corretamente, independentemente de atravessar o limiar de 30.

### 8.8 Protocolo de validação e organização das execuções

- **Folds agrupados por ciclone** (`dividir_folds`, mesmo algoritmo do `GroupKFold`): ordena os ciclones do mais para o menos janelas e atribui cada um ao fold com menos janelas até então (empates sorteados com `seed_estrutural`). Resultado: folds balanceados (≈ 13.134 janelas cada); um `assert` confirma que nenhum ciclone aparece em dois folds. O grupo é o `id_ciclone` **já unificado** pela detecção de duplicatas. A divisão é **fixa** (igual para todas as variantes e seeds), então todas são avaliadas **exatamente nas mesmas janelas** (comparação pareada).
- **Validação interna (para o *early stopping*):** do treino de cada fold separam-se 10% dos **ciclones** (inteiros); treina-se nos outros 90%.
- **Ordem dos laços:** fold → seed → variante, para que todas as variantes completem o mesmo (fold, seed) antes do próximo (comparações pareadas possíveis mesmo que o tempo acabe no meio).
- **Salvamento imediato:** cada execução vira `runs/<variante>__f<fold>__s<seed>__<assinatura>.npz` (índices das janelas de teste, rótulos, predições, previsões de ΔVmax da validação interna, histórico, metadados); na 1ª seed de cada fold grava-se também o `.pt` com os pesos (usados na explicabilidade).
- **Retomada:** execuções cujo arquivo existe são puladas; `cfg.dirs_retomada` aceita a pasta de saída de uma sessão anterior (anexada como *input* no Kaggle).
- **Assinatura:** hash curto dos hiperparâmetros que mudam o resultado e do conjunto de janelas, no nome do arquivo. Mudou hiperparâmetro, duplicatas ou dados → a assinatura muda → execuções antigas **não** são reaproveitadas por engano. Ficam **fora** dela os campos que não mudam o resultado (`_CAMPOS_SO_DE_EXECUCAO`): pastas, modo, variantes/folds/seeds a executar, parâmetros de estatística e explicabilidade e, desde 2026-10-12, `max_horas_treino`, `usar_cache`, `permitir_cpu`, `tam_lote_aval` e `log_por_epoca`. Antes, o orçamento de 10 h que o notebook liga no Kaggle (`max_horas_treino`) entrava na assinatura e mudava o nome das execuções: um treino feito em outra máquina (`treinar_remoto.py`) não seria reconhecido pelo notebook (testado: as assinaturas divergiam com a regra antiga e coincidem com a nova). Mudar a regra não afeta nada já gravado de valor: a 1ª execução do Kaggle já tinha outra assinatura (janelas diferentes, sem remoção de duplicatas).
- **Orçamento de tempo** (`cfg.max_horas_treino`): a sessão do Kaggle é limitada (≈ 12 h); por padrão **no Kaggle o notebook para de iniciar novos treinos depois de 10 h** (ou quando a próxima não couber) e segue para as tabelas e figuras com o que concluiu; o log avisa quantas ficaram pendentes (rodar de novo, ou numa nova sessão com `dirs_retomada`, completa o plano). Fora do Kaggle não há limite.

### 8.9 Estatística: três níveis de comparação

A **variante de referência** é a proposta (`cnn_lstm_kan`); cada outra variante (e, para ΔVmax, o `baseline_historico` sem imagens) é comparada com ela. Diferença **negativa** = a referência erra menos.

| Nível | Unidade do teste | Papel | Pontos fortes / fracos |
|---|---|---|---|
| **1: por janela** | em cada (fold, seed), Wilcoxon sobre os erros absolutos janela a janela (uma família de Holm) | apoio | muito poder, mas janelas do mesmo ciclone são correlacionadas (não são amostras independentes): p-valores *otimistas* |
| **2: por execução** | Wilcoxon sobre a métrica global (RMSE/MAE) de cada execução, pares = mesmo (fold, seed) | **exploratório** | as 5 seeds de um fold testam as mesmas janelas: os 25 pares são ≈ 5 observações (pseudorreplicação) e os p saem inflados |
| **3: bootstrap por ciclone** | reamostra **ciclones** com reposição; diferença de MAE/RMSE entre variantes nas mesmas janelas (erro médio entre seeds); IC 95% e p bicaudal; Holm sobre todos os testes | **principal** (notebook 7.3b) | a unidade independente é o ciclone (≈ 1,4 mil); embute a variação entre folds e entre ciclones |

- **Apoio ao nível 3: 1 ponto por fold.** A métrica de cada (variante, fold) é a média das seeds; compara-se com *t* pareado e Wilcoxon exato sobre os folds, e conta-se em quantos folds a referência erra menos (`folds_ref_melhor`). Com 5 pontos o menor p do Wilcoxon exato é $2/2^5=0{,}0625$: serve para ver se a **direção** se repete, não para declarar significância.
- **Como ler:** convincente quando o **IC 95% do bootstrap por ciclone não inclui 0** (p ajustado por Holm abaixo de α) e a direção se repete nos folds. As tabelas do nível 1 trazem, como a Tabela 5 do TCC, a média dos erros e a % de janelas em que a referência errou menos; o "veredito" diz *referência melhor / sem diferença / referência pior* depois do Holm.
- **Pseudorreplicação** (Hurlbert, 1984) é tratar como independentes medidas que não o são. Cada fold é um teste diferente, mas as 5 seeds de um fold avaliam as **mesmas janelas** (a seed só muda inicialização, ordem dos lotes e rotações). Na 1ª execução o desvio **entre folds** foi 0,47 nó e **entre seeds** 0,08 nó; o teste por execução (25 pares) deu "KAN melhor em MAE de Vmax" com p ajustado 0,035, e com 1 ponto por fold o p é 0,13. Repetir seeds serve para medir o ruído do treino (o bootstrap já o embute), não para multiplicar observações.
- **Bootstrap por ciclone, passo a passo:** (1) para cada janela de teste, o erro (|e| e e²) é a **média entre as seeds**; (2) somam-se, por ciclone, n, Σ|e| e Σe² de cada variante; (3) repete-se `cfg.bootstrap_reps` (2000) vezes: sorteiam-se N ciclones com reposição (pesos multinomiais), recalculam-se MAE e RMSE (= √(Σe²/n)) e a diferença referência − outra; (4) IC 95% = percentis 2,5 e 97,5; p bicaudal = 2·min(P(d ≥ 0), P(d ≤ 0)) com correção +1; Holm sobre todos os testes (comparações × tarefa × métrica). O RMSE do bootstrap é √(erro quadrático médio entre seeds), que pode diferir de milésimos da média dos RMSE por execução. Referência: Efron e Tibshirani (1993).
- Arquivos: `tabelas/wilcoxon_por_janela.csv`, `wilcoxon_por_execucao.csv`, `comparacao_por_fold.csv`, `bootstrap_por_ciclone.csv`.

### 8.10 Explicabilidade (implementação)

Ambos os métodos usam os pesos gravados da 1ª seed do primeiro fold executado (`cfg.variante_referencia`).

- **Grad-CAM** (notebook 8.1): mapas $A^k$ do último bloco convolucional do encoder (256 mapas de 4×4); peso por canal $\alpha_k=\frac1{HW}\sum_{h,w}\partial y/\partial A^k_{hw}$; $\mathrm{CAM}=\mathrm{ReLU}(\sum_k\alpha_kA^k)$; ampliado a 64×64 sobre o IR. Como o encoder processa cada quadro, obtém-se **um mapa por instante** (`t−12h … t`): mostra *quando* e *onde* o modelo prestou atenção. O *backward* de um lote inteiro (`saida.sum().backward()`) dá o gradiente correto de cada amostra porque o BatchNorm em modo de avaliação não mistura amostras. Figuras: `05_gradcam_exemplos_*.png` (3 janelas: Vmax ≈ 30, 70 e 120 nós, como no SHAP do TCC) e `06_gradcam_medio_*.png` (mapa médio sobre `gradcam_amostras` janelas de teste). A camada 4×4 é grosseira: ler como "qual região/quadrante", não como contorno. No teste de calibração (poucas épocas, subconjunto), o mapa de `t` se concentrava perto do olho nos ciclones intensos, coerente com o SHAP do TCC; refazer a leitura com um modelo bem treinado.
- **Splines da KAN** (notebook 8.2): na 1ª camada de cada cabeça, $u_j=\sum_i\varphi_{0,j,i}(h_i)$, onde $h_i$ é a $i$-ésima variável latente. **Importância:** a "importância de aresta" do artigo da KAN, a média de $|\varphi(h_i)|$ nos dados, propagada da saída para a entrada ponderando cada aresta pela importância do nó para o qual aponta (versão simplificada do *attribution score*). Figuras: `07_kan_importancia_*.png` (as 20 variáveis mais importantes) e `08_kan_splines_*.png` (linha de cima: a spline da aresta mais forte de cada uma das `spline_topk` variáveis principais, sobre o histograma dos valores que a variável assume; linha de baixo: a função da 2ª camada do nó oculto envolvido). As variáveis latentes da LSTM não têm nome físico ("h₁₇"): interpreta-se a **forma** da curva (crescente, decrescente, saturando, com degrau), quais variáveis dominam cada tarefa e se as tarefas as compartilham; uma curva só informa onde há dados (por isso o histograma ao fundo).
- **Diagnósticos adicionais da KAN** (notebook 8.2 e 8.3): (a) **esparsidade:** quanto da importância total as 10 variáveis principais concentram (uniforme seria 10/128 ≈ 7,8%) e o **número efetivo de variáveis** ($1/\sum p^2$; 128 = totalmente espalhada); uma KAN "interpretável" concentraria a importância em poucas variáveis; (b) **sobreposição entre tarefas × acaso:** nº de variáveis comuns às 10 principais de Vmax e de ΔVmax contra o esperado ao acaso (10·10/128 ≈ 0,8) e a probabilidade hipergeométrica P(≥ observado | acaso); (c) **significado das variáveis latentes:** uma $h_i$ é uma coordenada arbitrária do espaço da LSTM (girar o espaço daria outra rede igualmente boa), então o notebook calcula a **correlação de Spearman** de cada $h_i$ com Vmax e ΔVmax reais e com resumos do quadro `t` (temperatura IR média, desvio e mínimo; fração de pixels com IR < 220 K; PMW médio) e mostra, para cada grandeza, a variável latente que mais se correlaciona. Correlações baixas em todas as variáveis importantes significam que as splines descrevem uma representação ainda sem nome (e que a esparsificação da KAN, não implementada, seria o próximo passo).
- **Comparação de método com o TCC:** seção 6.7.

## 9. Resultados da 1ª execução no Kaggle (2026-10-12)

### 9.1 O que rodou

- **75 treinos** (5 folds × 5 seeds × 3 variantes) numa **T4**, em ≈ **4 h 31** (14:29 → 19:00 no relógio do Kaggle). Tempo médio por treino: 2 min 33 s (`cnn_kan_1frame`), 3 min 53 s (`cnn_lstm_mlp`), 4 min 24 s (`cnn_lstm_kan`; ≈ 10,9 s/época). Sem erros nem avisos relevantes.
- **Versão anterior do notebook**, sem a detecção interna de duplicatas (o log registra `nenhum duplicatas_tcir.csv encontrado — seguindo SEM remover duplicatas`): **65.730 janelas e 1.375 ciclones** (com a remoção seriam 65.670 e 1.373), 55.121 janelas com rótulo de ΔVmax, 3.674 de RI (6,7%), Vmax médio 51,77 (dp 28,89). As janelas e os folds foram **reproduzidos localmente** e a tabela de folds confere com a do log (274/275/275/276/275 ciclones; 13.146 janelas em cada fold; mesmos RI por fold).
- A variante `cnn_lstm_kan_vmax` (ablação da multitarefa) **não estava** na lista padrão e **não foi rodada**.
- Fonte dos números: log do notebook baixado (`kanscnn (3).ipynb`) e `analisar_resultados_kaggle.py`.

### 9.2 Resultados

Média ± desvio entre as 25 execuções (fold × seed); erros em nós.

| Variante | Vmax RMSE | Vmax MAE | ΔVmax RMSE | ΔVmax MAE | RI POD / FAR / CSI (limiar 30) |
|---|---|---|---|---|---|
| `cnn_lstm_kan` (proposta) | 9,58 ± 0,43 | 7,08 ± 0,18 | 15,87 ± 0,21 | 11,17 ± 0,14 | 0,16 / 0,57 / 0,13 |
| `cnn_lstm_mlp` | 9,63 ± 0,43 | 7,13 ± 0,18 | 15,98 ± 0,27 | 11,26 ± 0,15 | 0,15 / 0,57 / 0,13 |
| `cnn_kan_1frame` | 11,29 ± 0,44 | 8,27 ± 0,17 | 16,30 ± 0,24 | 11,50 ± 0,11 | 0,12 / 0,56 / 0,10 |

RMSE de Vmax por fold (média das 5 seeds): **o fold pesa muito mais que a seed**.

| Variante | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| `cnn_lstm_kan` | 9,40 | 9,76 | 9,21 | 9,21 | 10,32 |
| `cnn_lstm_mlp` | 9,42 | 9,88 | 9,22 | 9,33 | 10,31 |
| `cnn_kan_1frame` | 10,84 | 11,30 | 10,90 | 11,38 | 12,03 |

Desvio entre folds (`cnn_lstm_kan`): **0,47 nó**; entre seeds dentro de um fold: **0,08 nó**. Só nas janelas de ciclones de 2017 (N = 3.967; predições de uma seed; protocolo diferente do TCC): `cnn_lstm_kan` 8,35 / 6,41 (RMSE / MAE), `cnn_lstm_mlp` 8,39 / 6,43, `cnn_kan_1frame` 9,88 / 7,45, contra 10,06 / 7,53 do CNN-TC e 9,14 / 7,00 do MobileNetV2+SDTA no TCC.

### 9.3 O que os resultados sustentam (e o que não)

1. **A LSTM ajuda (conclusão robusta).** `cnn_lstm_kan` × `cnn_kan_1frame`: −1,71 nó de RMSE de Vmax (−15%), a proposta melhor em **5/5 folds e 25/25 execuções**; em ΔVmax, −0,43 nó (5/5 folds, 23/25 execuções). A variante de 1 quadro chega a 9,88 nas janelas de 2017, no nível do CNN-TC do TCC (10,06): o encoder está compatível e o ganho vem da sequência.
2. **KAN e MLP empatam.** Diferença de RMSE de Vmax de −0,05 nó (−0,6%); com 1 ponto por fold (n = 5) a proposta é melhor em 4/5 folds e o p do *t* pareado é 0,12 (MAE: 0,13; RMSE de ΔVmax: 0,08, melhor em 5/5 folds, −0,11 nó). O "KAN melhor em MAE de Vmax" (p ajustado 0,035) do teste por execução **não se sustenta** (seção 9.4, item 1). As cabeças são só ≈ 3% dos parâmetros (37 mil de 1,08 milhão), então esta ablação tem pouca alavanca. O argumento a favor da KAN, hoje, é **interpretabilidade, não acurácia**.
3. **O ΔVmax previsto pela rede não supera um baseline sem imagens.** Regressão polinomial de grau 2 em Vmax(t) e na tendência de 3, 6 e 12 h do *best track* (sem imagens) dá RMSE de ΔVmax **15,71**, contra **15,87** da rede (prever 0 dá 19,72; só a faixa de Vmax(t) real dá 18,65). A rede é pior em 4 dos 5 folds (diferenças por fold: +0,40, −0,12, +0,07, +0,01, +0,42). Ressalva: o baseline usa o histórico de intensidade, que a rede não recebe (é informação disponível a um previsor operacional, mas não ao modelo proposto). As imagens, portanto, **não estão acrescentando ao ΔVmax** nada além da tendência recente de intensidade, ao menos com esta arquitetura e protocolo.
4. **RI: a comparação ingênua enganava.** No limiar de 30 nós a rede tem POD 0,16 e o baseline de histórico 0,01, mas isso é efeito do limiar (a regressão por MSE encolhe as previsões: a rede sinaliza ≈ 2,5% das janelas, estimativa a partir de POD, FAR e prevalência, e o baseline 0,16%, quando 6,7% são eventos). Com limiar ajustado (no treino), o baseline de histórico chega a POD 0,39, FAR 0,75, CSI 0,18, **AUC 0,80 e AP 0,24** (prevalência 0,067): é o piso que a rede precisa bater.
5. **Vazamento por duplicata: real, mas desprezível.** Sem a remoção, o Otto ficou com `201616L` no fold 0 e `201622E` no fold 3: **65 janelas idênticas** separadas entre treino e teste (os pares `201010L`/`201011E` caíram nos folds 0 e 2, mas sem janelas no mesmo instante). São ≈ 0,2% das avaliações; mesmo com erro zero, o RMSE mudaria no máximo ≈ 0,01 nó. Não invalida a execução, mas o número final deve vir do notebook atual.

### 9.4 Problemas encontrados

1. **Pseudorreplicação no teste "por execução".** As 5 seeds de um fold avaliam as mesmas janelas e a variação entre folds é ≈ 6× a variação entre seeds: os 25 pares são ≈ 5 observações. O Wilcoxon exato com 5 pares tem p mínimo $2/2^5=0{,}0625$, que **nunca** passa de 0,05; logo o p de 0,035 era artefato do número de pares. O teste principal passou a ser o bootstrap por ciclone (seção 10.1).
2. **Faixa horizontal no gráfico predito × real.** Janelas com Vmax real entre ≈ 40 e 140 nós previstas em ≈ 20–28 nós **pelos três modelos**: aponta para a entrada ou o rótulo, não para a arquitetura. Não foi identificado quais janelas são (o `runs/*.npz` do Kaggle não foi baixado). **Pista (hipótese, não causa confirmada):** no teste do notebook ajustado (36 ciclones, 2 épocas, sem valor como resultado), as 15 piores janelas vieram todas do ciclone `201407E` (ago/2014, bacia EPAC; Vmax real 115–140 nós, previstos em 26–55, a maioria ≈ 26, pelas 4 variantes). Nas datas críticas (7–9/ago) ele estava em **longitude −179,8° a −175,9°, isto é, sobre a linha de data**, e as imagens IR dessas janelas **não mostram um ciclone intenso centrado** (sem olho; massa de nuvens deslocada para um canto; 0% de IR ausente). O recorte de 201×201 px (≈ 14°) de um ciclone assim cruza a emenda ±180° da grade de longitude. No TCIR completo há 1.318 imagens (68 ciclones) com |lon| > 175°, ou 1.129 janelas (1,7%), 395 delas com Vmax ≥ 60 nós. Falta confirmar, com as predições reais, se a faixa é feita dessas janelas.
3. **Sobreajuste moderado.** A perda de treino cai de 0,43 (época 5) para ≈ 0,19 (época 25) enquanto o RMSE de validação estagna em ≈ 9,7 (a perda de treino é medida com *dropout* e rotação, que a deixam *maior*, então a folga é real). Não tratado.
4. **Grad-CAM de baixa resolução.** Mapas 4×4 ampliados; nos exemplos de Vmax ≈ 30 e 70 nós o calor aparece nas bordas e nos quadros mais antigos, e não no núcleo do ciclone (no de 120 nós ele destaca o olho em `t−6h`). Leitura visual, sem teste de sanidade: tratar como indicativo.
5. **KAN sem estrutura esparsa.** A importância das 20 variáveis latentes mais importantes varia só de ≈ 0,057 a ≈ 0,037; as splines são suaves e quase monótonas. Das 10 mais importantes de cada tarefa só 2 coincidem (h17, h115): ao acaso se esperaria 0,8 (P(≥ 2 | acaso) ≈ 0,18), então **não há evidência de representação compartilhada** entre as tarefas; e a ablação só-Vmax, que testaria o benefício da multitarefa, não foi rodada.
6. **Comparação com o TCC não é direta:** split por ano × validação cruzada por ciclone.

## 10. Ajustes posteriores, verificação e próxima execução

### 10.1 Ajustes feitos no notebook depois dos resultados

| # | Problema | Ajuste | Onde (notebook) | Saídas novas |
|---|---|---|---|---|
| 1 | ablação da multitarefa não rodada | `cnn_lstm_kan_vmax` entra em `cfg.variantes` (padrão: 4 variantes) | célula de configuração (1) | |
| 2 | pseudorreplicação | teste **principal** = **bootstrap por ciclone** (`comparar_bootstrap`); apoio = **1 ponto por fold** (`comparar_por_fold`); o por execução fica como *exploratório* | 7.3b (+ texto da 7.3) | `tabelas/bootstrap_por_ciclone.csv`, `comparacao_por_fold.csv` |
| 3 | ΔVmax sem baseline | `baselines_dvmax`: zero, média do treino e histórico de Vmax (polinômio de grau 2), nos mesmos folds; o de histórico entra também no bootstrap (só ΔVmax) | 7.2b | `tabelas/baselines_dvmax_por_fold.csv`, `dvmax_modelos_vs_baselines.csv` |
| 4 | RI medido só com limiar 30 | `metricas_ri_todas`: limiar 30; **limiar ajustado na validação interna** (maximiza o CSI; nunca olha o teste); **AUC-ROC e AP**. Cada execução grava `y_dvmax_val` e `pred_dvmax_val` | 5.2, 5.3, 6.3, 7.2b | colunas `limiar_ri_aj`, `pod/far/csi_ri_aj`, `auc_ri`, `ap_ri` em `metricas_por_execucao.csv` |
| 5 | faixa de erro grande | diagnóstico de janelas com \|erro de Vmax\| ≥ `cfg.outlier_erro_nos` (40): fração do erro quadrático, o mesmo erro nas outras variantes, resumo da imagem (IR média/desvio/mínimo, fração fria, PMW, IR ausente) × demais janelas, 8 piores desenhadas | 7.5 | `tabelas/janelas_erro_grande_vmax.csv`, `figuras/09_janelas_erro_grande.png` |
| 6 | KAN: esparsidade e significado | concentração da importância, sobreposição contra o acaso (hipergeométrica), correlação de Spearman das latentes com grandezas físicas | 8.2, 8.3 | `tabelas/kan_latentes_correlacoes_<variante>.csv` |
| 7 | usuário não quer subir `duplicatas_tcir.csv` no Kaggle | o notebook **detecta e remove as duplicatas sozinho** (2.2; `cfg.detectar_duplicatas`, `tol_duplicata_graus = 0,25`); resultado idêntico ao do script (72 registros, 2 IDs); `verificar_duplicatas.py` virou só auditoria | 2.2 | |
| 8 | treinar numa máquina própria com GPU (RTX 5090), sem Kaggle | `treinar_remoto.py` (seções 1 a 6 do notebook como script) + `verificar_equivalencia_treino.py`; no notebook, só 3 mudanças: os 5 campos operacionais saem da assinatura (célula do protocolo, `_CAMPOS_SO_DE_EXECUCAO`) e duas notas de texto (célula inicial e 6.2). **O modelo, o treino e os dados não mudaram.** | 6 | `log_treino.txt`, `ambiente_treino.json` (só do script) |

Os campos novos de `Config` (`bootstrap_reps = 2000`, `baseline_sem_imagens = True`, `outlier_erro_nos = 40`) **não entram na assinatura** das execuções: mudá-los não invalida treinos já feitos. O `resumo_final.json` passa a incluir o baseline e o bootstrap. Execuções gravadas por versões anteriores não têm as previsões de validação: para elas as colunas `_aj` ficam vazias (AUC/AP continuam).

### 10.2 Não implementado de propósito (muda o modelo proposto; só com decisão explícita do usuário)

- **Histórico de Vmax como entrada da rede** (a proposta é "só imagens"; ver seção 12, item 12).
- **Cabeça de classificação para RI** (BCE com peso na classe rara, ou *focal loss*). O limiar ajustado e a AUC/AP são o que cabe sem mudar o modelo.
- **Esparsificação e poda da KAN** (regularização L1 + entropia, como no artigo original). As seções 8.2/8.3 do notebook apenas **medem** a falta de esparsidade e tentam dar significado físico às variáveis.
- **Experimento com o split por ano do TCC** (comparação estrita) e **tratamento do sobreajuste** (mais regularização/aumento de dados).
- A **causa da faixa de erro grande** só pode ser confirmada rodando a seção 7.5 do notebook sobre as predições reais.

### 10.3 Como reproduzir os números da seção 9

```bash
python3 -I analisar_resultados_kaggle.py --notebook "/caminho/kanscnn (3).ipynb"
```

`-I` porque o notebook baixado é um arquivo não confiável (o script só o lê como JSON). Depende de `estatisticas_janelas.py` e do cache `resultados_janelas/frac_invalido_ir.csv`. Saídas em `resultados_kaggle/`: `analise_kaggle.json` (todos os números) e `metricas_por_execucao_log.csv` (os 75 treinos extraídos do log, com 2 casas decimais). O script: **(A)** decompõe a variância em fold × seed e refaz as comparações com 1 ponto por fold; **(B)** reproduz as janelas/folds **sem remoção de duplicatas** (como o notebook rodou) e conta as janelas gêmeas separadas entre folds; **(C)** calcula os baselines de ΔVmax (zero, média do treino, Vmax(t) real por faixa de 10 nós, e Vmax(t) + tendência de 3/6/12 h em polinômio de grau 2; para o de histórico, também o RI com limiar ajustado no treino e AUC/AP); **(D)** conta as janelas perto da linha de data (|lon| > 175°) e descreve a posição do `201407E`. Os números de desempenho das redes vêm direto das tabelas do log do notebook.

### 10.4 Como foi verificado

Nenhuma etapa foi dada como pronta só por rodar sem erro:

| O que | Como | Resultado |
|---|---|---|
| B-splines da KAN | contra `scipy.interpolate.BSpline.design_matrix` (4 combinações de grid e grau) | diferença máx. ≈ 1e-7; somam 1 no grid |
| Soma das arestas da KAN | $\sum_i\varphi_{j,i}(x_i)$ == saída da camada; curva isolada == aresta | confere |
| Rotação | ângulo 0 ≈ recorte central; 90° e 180° == `torch.rot90`; 20 canais giram pelo mesmo ângulo; entrada constante em 73 ângulos de 0 a 360° | sem cantos vazios; erro só de ponto flutuante (≈ 4e-5) |
| DTA / SDTA | recalculados à mão, quadro a quadro | idênticos |
| Janelas | série sintética com buraco de 6 h, quadro inválido e ciclone curto | nenhuma janela atravessa buraco ou contém quadro inválido; Vmax(t) = rótulo do último dos 7 quadros; ΔVmax correto |
| Folds | 60 ciclones sintéticos | nenhum ciclone em 2 folds; balanceados; determinísticos |
| Holm-Bonferroni | contra `statsmodels.multipletests(method='holm')` | idêntico |
| Gradiente até o encoder | `backward` em todas as variantes | chega e é finito |
| Versões | funções de dados com pandas 2.3 e 3.0, NumPy 2.x, Matplotlib 3.11 | corrigidos `read_hdf`, `Timedelta`, `boxplot(labels=…)` |
| Sintaxe para o Kaggle | células de código e `verificar_duplicatas.py` compilados com **Python 3.11** (o venv de teste era 3.12, que aceita construções que o 3.10/3.11 recusam) | compila (também em 3.12 e 3.13) |
| Notebook de verdade | executado por kernel Jupyter (`nbconvert --execute`) em modo teste com dados reais | 0 erros; retomada pulou as execuções já feitas; unificação de IDs exercitada (38 IDs → 36 ciclones) |
| Detecção de duplicatas no notebook | funções executadas nos dados completos e comparadas ao CSV do script | mesmos 72 registros e mesmos IDs unificados; 70/72 idênticos nos pixels; 65.670 janelas, 549 quadros inválidos, 3.669 RI |
| Duplicatas (script) | executado nos 48 GB reais, duas vezes (rápido e com hash) | seção 7.3 |
| Funções novas dos ajustes | AUC/AP contra força bruta; bootstrap contra o cálculo direto (idênticos até 1e-6) e com verdade conhecida (IC exclui 0 quando uma variante é melhor e o contém quando têm o mesmo ruído); limiar ajustado numa previsão encolhida pela metade (acha ≈ 12 nós, POD de 0,00 para 1,00); baseline do notebook × script salvo sobre as 65.730 janelas (RMSE de ΔVmax 19,723 / 19,720 / 15,712 para zero / média / histórico, diferença ≤ 0,005) | passou |
| Notebook ajustado inteiro | PyTorch 2.14 em CPU, Python 3.13, `data/20percent/` (36 ciclones; 16 execuções = 2 folds × 2 seeds × 4 variantes; 2 épocas; ≈ 8 min) | todas as células sem erro; 15 figuras e 10 tabelas, inclusive as novas; o diagnóstico de janelas já apontou a pista da linha de data |
| `treinar_remoto.py` × notebook (equivalência) | `verificar_equivalencia_treino.py`: árvore sintática de cada definição; teste negativo (mudei `lr` no script) | 38 definições idênticas, 2 diferenças declaradas; o teste negativo falhou como devia (código 1) |
| `treinar_remoto.py` ponta a ponta | modo teste em CPU, `data/20percent/` (mesmos 16 treinos de cima), de um diretório vazio; depois o notebook (como script, CPU) apontado para a **mesma saída** | script: 16 execuções, código 0, 1.083.104 / 1.062.594 / 967.520 / 1.064.528 parâmetros (iguais à seção 8.5); notebook: `já concluídas: 16 \| pendentes: 0`, carregou as 16 e gerou tabelas e figuras sem erro (código 0). Mesma seed ⇒ mesmo RMSE do 1º treino em duas rodadas na CPU (22,62) |
| Assinatura entre máquinas | `assinatura_experimento` com `max_horas_treino` = 10 (Kaggle) × sem limite (remoto) e `tam_lote_aval`/`usar_cache` diferentes | regra antiga: diferente; regra nova: **igual**; `epocas_max` ou `tam_lote` diferentes continuam mudando a assinatura |
| Robustez do script | retomada (mesmo comando de novo); `SIGTERM` no meio de um treino; 2 processos com folds diferentes na mesma saída; modo completo sem GPU; modo completo com a pasta de 20%; `--dispositivo cuda` sem GPU; sem `--dados` | retomada pula tudo; SIGTERM sai com 130 sem arquivo truncado e a retomada continua; 2 processos sem colisão; as 4 travas recusam com mensagem clara |
| Sintaxe do script | `treinar_remoto.py` compilado com Python 3.11 e 3.13; `pyflakes` | compila; sem avisos. As 30 células de código do notebook também compilam em 3.11 |

**Não verificado:** GPU/Kaggle e dados completos (65 mil janelas) com o notebook ajustado, e as células de treino após a mudança da detecção interna de duplicatas (não alteram nada além de `cfg`). A primeira execução real do notebook ajustado é, portanto, também o primeiro teste em escala; vale rodar antes o modo teste no Kaggle. **Também não verificado: `treinar_remoto.py` em GPU CUDA** (só CPU), em particular na RTX 5090 (Blackwell); e o tempo por execução nela (não medido).

### 10.5 Teste de sanidade de aprendizado (escala reduzida; **não são resultados do experimento**)

250 ciclones (≈ 9,5 mil janelas de treino), fold 0, **1 seed**, 8 épocas, GPU Apple. Uma execução com ≈ 15% dos dados não sustenta conclusão sobre as variantes; só mostra que o código aprende e que o ranking é plausível. Para escala: o CNN-TC do TCC teve RMSE 10,06 com ≈ 60 mil imagens; um preditor constante teria ≈ 29.

| Variante | Vmax RMSE | Vmax MAE | ΔVmax RMSE |
|---|---|---|---|
| `cnn_lstm_kan` | 10,93 | 8,28 | 15,47 |
| `cnn_lstm_mlp` | 11,31 | 8,58 | 15,76 |
| `cnn_kan_1frame` | 13,13 | 9,99 | 16,59 |
| `cnn_lstm_kan_vmax` | 10,53 | 8,03 | — |

### 10.6 O que esperar da próxima execução

- **100 treinos** (5 × 5 × 4 variantes): ≈ 4,5 h dos 75 anteriores + ≈ 25 treinos só-Vmax (≈ 4 min cada, estimativa) ≈ **6–6,5 h**, dentro do orçamento de 10 h. Como `variantes` não entra na assinatura, dá para rodar antes só as 3 variantes e acrescentar a 4ª depois.
- Janelas e folds mudam levemente (65.670 janelas, 1.373 ciclones): os números **não** serão idênticos aos da 1ª execução; o que se compara é a conclusão.
- Ordem de leitura: seção 2, item 3.

## 11. Decisões tomadas e desvios da especificação / do TCC

| # | Tema | Especificação / TCC | Implementado | Motivo |
|---|---|---|---|---|
| 1 | Pmin | 3 cabeças | 2 cabeças (Vmax, ΔVmax) | decisão do usuário; `MSLP` já existe se quiser voltar |
| 2 | Escopo da remoção de duplicatas | só no treino | **todos os dados** + unificação de IDs | a janela exige instantes únicos; cópias não podem cair em folds diferentes |
| 3 | Onde detectar duplicatas | etapa da pipeline | dentro do notebook (2.2); o script só audita | o usuário não quer anexar CSV no Kaggle |
| 4 | Quadros com IR ausente | TCC: `NaN → 0` e segue | quadro com > 10% de IR ausente é descartado | 0,64% das imagens vazias carregariam rótulo sem informação; abrem buraco temporal |
| 5 | Pixels ausentes em quadro válido | `NaN → 0` antes do z-score | idem (padrão `"tcc"`), com opção `"mascara"` | fidelidade ao TCC + alternativa limpa |
| 6 | Quadros por janela | 5 | 5 na janela, **7 lidos** | DTA/SDTA do quadro mais antigo |
| 7 | Rotação | gira a imagem 201×201 e corta 64×64 | gira o recorte 92×92 e corta 64×64 | equivalente na região central, bem mais barato; sem cantos vazios |
| 8 | Probabilidade de rotação | "0,5 ou 0,55" | **0,55** | a confirmar |
| 9 | Framework | Keras/TensorFlow | **PyTorch** | KAN tem referências em PyTorch; vem no Kaggle |
| 10 | Encoder | CNN-TC / ResNet50 / MobileNetV2 | CNN-TC-like **treinada do zero** (+ BatchNorm, Dropout, LayerNorm) | roda 5× por amostra |
| 11 | KAN | `pykan` / `efficient-kan` | **implementação própria** `[128→16→1]`, grid 5, grau 3 | sem dependências, offline, comentada; validada contra SciPy |
| 12 | Modelos comparados | "vários modelos" | **4 variantes** que isolam KAN, LSTM e multitarefa (a 4ª entrou em 2026-10-12) | os testes precisam de ≥ 2 modelos |
| 13 | Validação | split por ano | **CV agrupada por ciclone (k = 5)**, todos os anos | prevista na especificação; ver seção 13 |
| 14 | Validação interna | `EarlyStopping` no conjunto de validação do ano | 10% dos ciclones do treino de cada fold | cada fold precisa da sua |
| 15 | Teste estatístico | Wilcoxon pareado sobre as 4.296 imagens | **bootstrap por ciclone** (principal) + 1 ponto por fold; Wilcoxon por janela (apoio) e por execução (exploratório); Holm em todos | janelas e seeds não são amostras independentes (seção 8.9) |
| 16 | Inferência | TCC usa *Rotation Blending* (média sobre rotações) | **sem** aumento em teste | a especificação limita a rotação ao treino |
| 17 | Alvos | Vmax em nós | Vmax e ΔVmax **padronizados** no treino | perdas em escalas comparáveis |
| 18 | Peso das tarefas | n/a | 1,0 (Vmax) e 0,5 (ΔVmax) | a principal pesa mais; ajustável |
| 19 | Armazenamento | `float32` | `float16` nos quadros | metade da memória; erro ≤ 0,125 K no IR, irrelevante frente às diferenças do DTA/SDTA (vários K) |
| 20 | Baseline sem imagens | n/a | zero, média e **histórico de Vmax** (poly 2) para ΔVmax | dizer que as imagens ajudam exige saber o que dá para prever sem elas |
| 21 | Limiar de RI | alarme em ΔVmax previsto ≥ 30 | limiar **ajustado na validação interna** + AUC/AP, além do 30 | a regressão por MSE encolhe as previsões; o 30 subestima a habilidade |
| 22 | Janelas com erro grande | n/a | diagnóstico (notebook 7.5) | faixa de previsões travadas em ≈ 25 nós nos 3 modelos |
| 23 | Interpretabilidade da KAN | splines + importância | + esparsidade, acaso na sobreposição e correlação das latentes; **sem** poda | medir antes de mudar o modelo proposto |
| 24 | Onde treinar | notebook no Kaggle | também `treinar_remoto.py` numa máquina com GPU (RTX 5090); mesmos hiperparâmetros padrão e mesmos arquivos de saída | o usuário tem uma máquina com essa GPU; o notebook continua sendo a referência (o script é cópia literal, conferida por `verificar_equivalencia_treino.py`) |
| 25 | Assinatura das execuções | todos os campos de `Config` fora de uma lista de exceções | + `max_horas_treino`, `usar_cache`, `permitir_cpu`, `tam_lote_aval`, `log_por_epoca` ficam de fora | não mudam o resultado; o orçamento de tempo do Kaggle impedia reconhecer treinos feitos em outra máquina |

## 12. Decisões em aberto (o usuário decide; apresente recomendação, não decida sozinho)

1. **Δ e limiar de RI:** implementado 24 h / 30 nós (padrão da literatura: Kaplan e DeMaria, 2003). Confirmar.
2. **Significado de "h(t)→Δ"** na cabeça de ΔVmax: implementado como horizonte do **alvo** (nenhum quadro futuro é entrada). Confirmar.
3. **Encoder:** CNN-TC-like do zero. Alternativa: ResNet50/MobileNetV2, inicializados com pesos do TCC (≈ 5× mais caro por amostra; trocar `CodificadorCNN`).
4. **Probabilidade de rotação:** 0,55 adotado; a alternativa é 0,5.
5. **Framework:** PyTorch (o TCC era Keras/TF). Confirmar a migração.
6. **Cabeças do artigo:** Vmax + ΔVmax. Reabrir Pmin só se o usuário pedir (`MSLP` já está no TCIR; custo baixo).
7. **Duplicatas:** remoção em todos os dados e unificação dos IDs `201622E → 201616L` e `201011E → 201010L`. Faz sentido? Vale mencionar no artigo que o TCIR (e portanto o TCC) tem 72 registros duplicados.
8. **Imagens vazias (0,64%):** descartadas; o TCC as manteve zeradas. O N = 4.296 do teste do TCC (contra 4.580 imagens de 2017) sugere um filtro: conferir no código do TCC o que foi feito.
9. **Comparação com o TCC:** reportar também um experimento com o split por ano (2003–14 / 2015–16 / 2017)?
10. **Ablações do artigo:** as 4 variantes atuais; outras possíveis (sem DTA/SDTA; 3 quadros em vez de 5).
11. **Perda:** pesos 1,0 / 0,5; tratar o RI como classificação (cabeça com BCE ou *focal loss*)?
12. **Histórico de Vmax como entrada:** se a rede continuar sem superar o baseline de histórico em ΔVmax, dar o Vmax passado (*best track*) como entrada adicional ou justificar a cabeça de ΔVmax de outro modo (p. ex., detecção de RI)? Hoje a proposta é "só imagens".
13. **A interpretabilidade da KAN é central no artigo?** Se sim: esparsificar/podar (L1 + entropia), grid adaptativo e dar nome físico às variáveis latentes.
14. **Sobreajuste:** mais regularização/aumento de dados?

Resolvida: a fonte do rótulo de Pmin (a coluna `MSLP` do TCIR).

## 13. Limitações e trabalhos futuros

- **Horizonte fixo e discreto.** A LSTM supõe passos de 3 h. Janelas com buracos são descartadas e quadros inválidos viram buracos: em uso operacional, um instante sem imagem não teria previsão. Neural ODE (tempo contínuo) é o caminho natural.
- **Comparação com o TCC não é direta.** O TCC treina em 2003–2014 e testa em 2017; aqui todos os anos entram na CV e as janelas exigem 7 quadros consecutivos. O notebook (7.2) mostra também as métricas só nas janelas de 2017, mas essas predições vêm de modelos que, em outros folds, treinaram com outros ciclones de 2017. Uma comparação estrita exigiria um experimento extra com o split por ano.
- **Testes por janela são otimistas** (janelas do mesmo ciclone são correlacionadas) e o teste por execução é pseudorreplicação; o veredito é o bootstrap por ciclone, com 1 ponto por fold como apoio.
- **Intensificação rápida é um evento raro** (6,7%): com perda quadrática o regressor tende à média e a POD fica baixa. Alternativas: perda ponderada, *focal loss* ou uma cabeça de classificação (fora do escopo desta versão). O notebook ajusta o limiar na validação e reporta AUC/AP, mas a cabeça continua sendo de regressão.
- **Grad-CAM é grosseiro** (mapas 4×4 ampliados) e heurístico. As variáveis latentes da LSTM não têm nome físico; as splines mostram a forma da relação, não um significado meteorológico.
- **KAN sem atualização de grid.** O grid das splines é fixo ((−1, 1) e (−3, 3)); o artigo da KAN propõe adaptá-lo aos dados. Sem regularização de esparsidade (L1/entropia) as curvas não são podadas, e na 1ª execução a importância ficou espalhada entre as variáveis latentes.
- **Rótulos do *best track*** têm incerteza própria (80% dos Vmax do TCIR são múltiplos de 5 nós) que limita o RMSE alcançável.
- **ΔVmax × imagens.** Na 1ª execução a rede não superou um baseline de histórico de Vmax; se isso se mantiver, é preciso decidir entre dar o histórico como entrada e justificar a cabeça de ΔVmax de outro modo.
- **Sobreajuste moderado** e **faixa de janelas com erro grande** ainda sem causa confirmada.
- **Pmin** fica de fora (seção 11, item 1).

## 14. Como rodar

### 14.1 Kaggle (caminho principal)

1. **Dataset com os 3 `TCIR-*.h5`** (≈ 48 GB; o mesmo usado no TCC serve). O notebook procura `*.h5` com "tcir" no nome em qualquer subpasta de `/kaggle/input`, então o nome do dataset não importa.
2. **Duplicatas: nada a fazer** (a detecção interna leva ≈ 1 s). Conferir no log `duplicatas: 72 registros a remover, 2 ID(s) unificado(s) com outro`.
3. *New Notebook → File → Import Notebook →* `pipeline_tc_kaggle.ipynb`; *Add Input* com o(s) dataset(s).
4. *Settings → Accelerator →* **GPU T4**. A internet pode ficar **desligada**: usa PyTorch, NumPy, pandas, SciPy, h5py e Matplotlib, que já vêm no Kaggle (`statsmodels` é opcional: só confere o Holm).
5. **Modo teste** (na célula de configuração, seção 1 do notebook: `MODO = "teste"`; *Run All*): percorre **todas** as etapas com 36 ciclones, 3 folds (2 executados), 2 seeds e 2 épocas, em minutos. Os números não valem nada (há aviso no log); serve para pegar problema de ambiente antes de gastar horas. A saída vai para `…/saida_pipeline/teste/`.
6. **Execução completa:** `MODO = "completo"` → *Save Version → Save & Run All (Commit)* (roda em segundo plano; o limite de uma sessão com GPU costuma ser ≈ 12 h). A cada treino concluído o log mostra RMSE/MAE, melhor época, duração e tempo restante estimado.
7. **Se a sessão cair ou o tempo acabar:** nada se perde (cada execução é gravada ao terminar). Rodar de novo pula as concluídas; ou, em outra sessão, anexar a saída anterior como *input* e `cfg.dirs_retomada = ("/kaggle/input/<saida-anterior>/saida_pipeline",)`.

### 14.2 O que é gerado (`/kaggle/working/saida_pipeline/`)

| Pasta/arquivo | Conteúdo |
|---|---|
| `config.json` | todos os parâmetros usados (reprodutibilidade) |
| `cache/` | quadros extraídos (`quadros_*.npy`, ≈ 2,5 GB); `cfg.usar_cache = False` evita carregá-lo na saída (a extração leva ≈ 1–5 min a cada sessão) |
| `runs/` | um `.npz` por execução (predições do fold de teste + histórico) e `.pt` (pesos, só da 1ª seed de cada fold) |
| `tabelas/metricas_por_execucao.csv`, `resumo_por_variante.csv` | uma linha por (variante, fold, seed); média ± desvio entre execuções |
| `tabelas/wilcoxon_por_janela.csv`, `wilcoxon_por_execucao.csv` | testes com p brutos e ajustados (Holm); o por execução é exploratório |
| `tabelas/bootstrap_por_ciclone.csv`, `comparacao_por_fold.csv` | **teste principal** (IC 95% da diferença de MAE/RMSE, p, veredito) e 1 ponto por fold |
| `tabelas/baselines_dvmax_por_fold.csv`, `dvmax_modelos_vs_baselines.csv` | baselines de ΔVmax sem imagens e a tabela redes × baselines (inclui RI) |
| `tabelas/janelas_erro_grande_vmax.csv` | janelas com \|erro de Vmax\| grande: erro em cada variante e resumo da imagem |
| `tabelas/kan_latentes_correlacoes_<variante>.csv` | correlação de Spearman das 128 variáveis latentes com grandezas físicas |
| `figuras/00…09_*.png` | distribuições, tensor (`01_*`), predito × real, RMSE por execução e curvas, erro por intensidade, Grad-CAM (`05_*`, `06_*`), importância (`07_*`) e splines (`08_*`) da KAN, janelas de maior erro (`09_*`) |
| `resumo_final.json` | resumo do experimento (inclui baseline e bootstrap) |

A última célula zipa **só `DIR_SAIDA`** (sem o `cache/`): é o `resultados_pipeline.zip` a baixar.

### 14.3 Parâmetros de `cfg` que costumam mudar (célula "Configuração", seção 1 do notebook)

| Parâmetro | Padrão | Para quê |
|---|---|---|
| `variantes` | as 4 | quais modelos treinar/comparar (tirar a 4ª economiza ≈ 1,7 h) |
| `seeds`, `folds_executar` | `(0, 1, 2, 3, 4)`, `None` (5 folds) | reduzir se faltar tempo (p. ex. `(0, 1, 2)`; `(0, 1)` roda só 2 folds) |
| `epocas_max`, `paciencia` | 25, 8 | duração do treino e *early stopping* |
| `max_horas_treino` | `None` (10 h no Kaggle) | orçamento para novos treinos (`0` desliga fora do Kaggle) |
| `tam_lote`, `tam_lote_aval` | 128, 256 | reduzir se faltar memória da GPU |
| `horizonte_horas`, `limiar_ri_nos` | 24, 30 | Δ e limiar da intensificação rápida (a confirmar) |
| `prob_rotacao` | 0,55 | chance de girar uma amostra de treino (a confirmar) |
| `tratamento_invalidos`, `max_frac_nan_ir` | `"tcc"`, 0,10 | pixels ausentes e descarte de quadros |
| `kan_ocultas`, `kan_grid`, `kan_ordem` | `(16,)`, 5, 3 | tamanho da cabeça; `kan_ocultas=()` dá uma KAN de 1 camada (modelo aditivo, ainda mais interpretável) |
| `pesos_tarefas` | `{"vmax": 1.0, "dvmax": 0.5}` | peso de cada tarefa na perda |
| `detectar_duplicatas`, `tol_duplicata_graus` | `True`, 0,25 | detecção interna de duplicatas |
| `bootstrap_reps`, `baseline_sem_imagens`, `outlier_erro_nos` | 2000, `True`, 40 | estatística e diagnósticos (200 réplicas no modo teste) |
| `variante_referencia`, `alfa` | `cnn_lstm_kan`, 0,05 | modelo comparado nos testes e significância |
| `usar_cache`, `dirs_retomada`, `permitir_cpu` | `True`, `()`, `False` | cache de quadros, retomada de outra sessão, permitir o modo completo sem GPU |

### 14.4 Tempo e recursos

- **Memória:** a GPU guarda os quadros (≈ 2,5 GB em `float16`) mais o modelo e os lotes: ≈ 4–6 GB, cabe numa P100/T4 (16 GB). Na RAM, ≈ 2,5 GB (≈ 5 GB durante a extração). Em disco, o cache (2,5 GB) e ≈ 5 MB por execução com pesos.
- **Tempo:** cada execução é `épocas × tempo por época`; um treino vê ≈ 47 mil janelas por época. Mac (GPU Apple/MPS, bem mais lenta que uma GPU de data center): ≈ 600 janelas/s, ≈ 80 s/época para a proposta. **Kaggle T4 (medido em 2026-10-12):** ≈ 10,9 s/época e 4 min 24 s por execução da `cnn_lstm_kan`; 9,6 s/época e 3 min 53 s da `cnn_lstm_mlp`; 6,1 s/época e 2 min 33 s da `cnn_kan_1frame` (o encoder roda 1 vez por amostra, não 5). As 75 execuções levaram 4 h 31; o plano de 100 treinos fica em ≈ 6–6,5 h. Se apertar: menos seeds, folds, épocas ou variantes.

### 14.5 Fora do Kaggle (Mac/Linux)

O notebook roda em qualquer lugar com PyTorch, NumPy, pandas, SciPy, h5py e Matplotlib. Só o modo teste vale em CPU (o completo recusa rodar sem GPU a não ser com `permitir_cpu=True`); em Mac com chip Apple o dispositivo é `mps` automaticamente. Com o disco cheio, usar `usar_cache=False`.

```bash
TCIR_DIR=/caminho/para/data  TCIR_SAIDA=./saida_pipeline  TCIR_MODO=teste \
    jupyter nbconvert --to notebook --execute pipeline_tc_kaggle.ipynb
```

### 14.6 Scripts auxiliares (sem PyTorch; precisam de `numpy`, `pandas`, `h5py`, `scipy`)

```bash
python3 verificar_duplicatas.py --dados data --saida resultados_duplicatas --varrer-imagens   # ≈ 100 s, lê 48 GB (sem --varrer-imagens: ≈ 5 s; --tol-graus muda a tolerância)
python3 estatisticas_janelas.py                                  # ≈ 30 s na 1ª vez (depois usa o cache); detecta duplicatas com a mesma função do notebook
python3 cortar_dataset.py                                        # data/ -> data/20percent/ (--frac 0.2 --seed 0 --compressao 4)
python3 -I analisar_resultados_kaggle.py --notebook "/caminho/kanscnn (N).ipynb"
```

### 14.7 Problemas comuns

| Sintoma | Causa e solução |
|---|---|
| `FileNotFoundError: Nenhum arquivo TCIR-*.h5` | dataset não anexado, ou ajustar `cfg.dir_dados` |
| `RuntimeError: Modo 'completo' sem GPU` | ligar o acelerador e reiniciar a sessão |
| `no kernel image is available for execution on the device` | PyTorch novo sem kernels para a arquitetura da P100 (Pascal): trocar para **T4**. Na **RTX série 50** (Blackwell, `sm_120`) é o oposto: PyTorch **antigo** (< 2.7 ou sem CUDA 12.8): `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128` (o `treinar_remoto.py` avisa na partida) |
| `CUDA out of memory` | reduzir `tam_lote` e `tam_lote_aval` |
| `ATENÇÃO: cfg.detectar_duplicatas = False` | detecção desligada; voltar para `True` (a menos que seja de propósito) |
| `AssertionError: há (ciclone, instante) repetido` | detecção desligada com dados que têm duplicatas estritas |
| treino "recomeça" do zero | mudou algum hiperparâmetro ⇒ nova assinatura; proposital, as antigas continuam em `runs/` |
| Wilcoxon por execução vazio | menos de 3 pares (fold, seed) em comum: rodar mais folds/seeds |
| `execuções ficaram PENDENTES` | o orçamento de tempo acabou antes de completar o plano: rodar de novo (retoma) ou aumentar `max_horas_treino` |

### 14.8 Treino numa máquina remota (RTX 5090): `treinar_remoto.py`

Roda só a parte de treino (seções 1 a 6 do notebook) e grava `runs/` com os mesmos nomes e conteúdo. A análise (tabelas, estatística, figuras) continua no notebook. Máquina informada pelo usuário: RTX 5090 e 32 GB de RAM (ele disse 16 GB de VRAM; a 5090 de mesa tem 32 GB e a de notebook 24 GB: o script imprime a VRAM real; o modelo usa ≈ 4–6 GB, então serve). Sem Jupyter, matplotlib nem scipy.

```bash
# preparo (uma vez; Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128     # RTX 50 (Blackwell, sm_120): PyTorch >= 2.7 com CUDA 12.8 e driver NVIDIA >= 570
pip install numpy pandas h5py
# 1) teste (minutos; confere GPU, dados e ambiente; números sem valor; saída em <saida>/teste/)
python treinar_remoto.py --dados /caminho/TCIR --saida /caminho/saida_pipeline --modo teste
# 2) experimento completo (100 execuções), sem depender do terminal
nohup python treinar_remoto.py --dados /caminho/TCIR --saida /caminho/saida_pipeline --modo completo > treino.out 2>&1 &
tail -f /caminho/saida_pipeline/log_treino.txt
# 3) caiu? o MESMO comando retoma (pula o que já está em runs/)
```

- **Dados:** os 3 `TCIR-*.h5` (≈ 48 GB) na máquina. `--dados` não deve conter `20percent/` (a busca é recursiva): no modo completo o script **recusa** um conjunto com menos de 75.079 imagens.
- **Opções úteis:** `--variantes`, `--folds`, `--seeds`, `--max-horas H` (para de iniciar treinos; sai com código 3), `--retomar-de DIR`, `--sem-cache`, `--dispositivo`, `--permitir-cpu`; vários GPUs: `CUDA_VISIBLE_DEVICES=1`. `--epocas` e `--lote` mudam o experimento (e a assinatura): só para testes. Defaults = os do notebook (lote 128, 25 épocas, AMP em `float16`, a mesma precisão da T4).
- **Se a GPU sobrar** (`nvidia-smi`; o modelo é pequeno): dois processos com folds diferentes na **mesma** saída (`--folds 0 1` e `--folds 2 3 4`); o 2º só depois de o 1º registrar `cache gravado` (testado em CPU).
- **Saídas** (além de `cache/` e `runs/`): `log_treino.txt` (acrescenta), `config.json`, `ambiente_treino.json` (GPU, versões, argumentos, duração, **pico de memória da GPU**). Códigos de saída: 0 completo, 3 pendentes (`--max-horas`), 130 interrompido.
- **Análise depois:** copiar `runs/` (≈ 130 MB; sem o cache) e rodar o notebook em modo completo apontando para a mesma pasta (`TCIR_SAIDA`, ou `cfg.dirs_retomada` no Kaggle com `runs/` anexado como dataset). Ele reconhece as execuções pela assinatura, pula o treino e gera tudo. Os `cfg` do notebook precisam ser os padrão (o mesmo que o script usou), senão a assinatura não bate e ele retreina. Testado só no modo teste, em CPU.
- **Tempo na 5090:** não medido (na T4: ≈ 4,4 min por execução da `cnn_lstm_kan`, ≈ 6–6,5 h os 100 treinos). O 1º `--modo completo` dá o número (o log mostra s/época e o tempo restante): registrar aqui.

## 15. Ambiente da máquina original (pode diferir em outra máquina)

- Mac com **9 GB de RAM e disco a 100%** (≈ 1,9 GB livres em 2026-10-02). O treino real é no **Kaggle** (GPU T4/P100; o TCC levou ≈ 4 h por modelo). Os 3 `.h5` (≈ 48 GB) ficam em `data/`.
- Não escrever caches grandes em disco (os quadros ocupam ≈ 2,5 GB) nem rodar o dataset completo em memória localmente (a escrita em disco já falhou uma vez): usar `usar_cache=False` e subconjuntos. Lotes grandes (≥ 1500 janelas) no MPS corromperam resultados silenciosamente por pressão de memória.
- **Armadilha de `data/20percent/`:** guarda cópias reduzidas dos `.h5`, e a busca do notebook (`rglob` ordenado, o primeiro nome vence) acharia as de 20% antes das completas com `dir_dados=data`. Apontar `TCIR_DIR` exatamente para a pasta desejada.
- Python do sistema 3.13 com pandas 3 e NumPy 2.x, **sem PyTorch**. `pd.read_hdf` falha no TCIR com pandas 3 + PyTables: ler o `info` direto com `h5py` (seção 7.2).
- Teste local com PyTorch: `/opt/homebrew/bin/python3.13 -m venv --system-site-packages <venv>` + `pip install --no-cache-dir torch` (≈ 700 MB; **apagar depois**, o disco fica no limite). Forçar CPU (trocar a checagem do MPS por `False` no texto do script) e rodar de um **diretório vazio**: uma versão antiga da última célula zipava o diretório atual (criou um `.zip` de 175 MB por engano e, na pasta do projeto, tentaria zipar `data/`); já corrigido para zipar só `DIR_SAIDA`. Modo teste com `TCIR_DIR=data/20percent` leva ≈ 8 min em CPU. Compilar o código também com Python 3.11 (versão do Kaggle).
- Shell macOS: sem `timeout` nem `sleep` longo; `sed -i ''` exige sufixo.
- **Máquina remota de treino (informada em 2026-10-12):** RTX 5090 (o usuário disse 16 GB de VRAM; conferir com `nvidia-smi`) e 32 GB de RAM; sistema operacional não informado (o script evita o que é só de Linux). É onde `treinar_remoto.py` deve rodar (seção 14.8). O disco dela precisa de ≈ 48 GB (dados) + 2,5 GB (cache) + ≈ 130 MB (`runs/`).

## 16. Glossário

| Termo | Significado |
|---|---|
| **TCIR** | *Tropical Cyclone for Image-to-Intensity Regression*: imagens de satélite de ciclones com a intensidade (Chen et al., 2018) |
| **IR / WV / VIS / PMW** | infravermelho / vapor d'água / luz visível / chuva por micro-ondas passivas |
| **Vmax** | velocidade máxima sustentada do vento (nós) |
| **MSLP / Pmin** | pressão central mínima (hPa) |
| ***Best track*** | séries oficiais de posição e intensidade dos ciclones; fonte dos rótulos de Vmax e MSLP no TCIR |
| **ΔVmax / RI** | variação de Vmax em 24 h / intensificação rápida (≥ 30 nós em 24 h) |
| **DTA / SDTA** | diferença temporal absoluta (1ª ordem) / segunda diferença (2ª ordem) entre quadros |
| **Quadro (*frame*)** | uma imagem da sequência de um ciclone |
| **Janela deslizante** | sequência de 5 quadros consecutivos que avança 3 h por vez |
| **Embedding / vetor latente** | resumo numérico de uma imagem produzido pelo encoder |
| **Pesos compartilhados** | a mesma rede aplicada a todos os quadros |
| **LSTM** | rede recorrente com portões e estado de célula; resolve o gradiente que some da RNN |
| **KAN** | rede de Kolmogorov-Arnold: funções aprendíveis nas arestas (B-splines), soma nos nós |
| **B-spline** | curva polinomial por partes, base das funções da KAN |
| **Multitarefa** | um modelo com tronco compartilhado e várias cabeças/saídas |
| **Ablação** | remover/trocar um componente para medir sua contribuição |
| **Grad-CAM** | mapa de calor por gradientes que mostra onde a CNN olhou |
| **Validação cruzada agrupada** | k-fold em que todas as amostras de um mesmo grupo (ciclone) ficam no mesmo fold |
| **Vazamento (*data leakage*)** | informação do teste "vazando" para o treino, inflando o desempenho |
| **Seed** | semente do gerador aleatório |
| **Wilcoxon pareado** | teste não paramétrico de postos sinalizados para amostras pareadas |
| **Holm-Bonferroni** | correção de comparações múltiplas que controla a taxa de erro familiar |
| ***Early stopping*** | parar o treino quando a validação deixa de melhorar |
| **BatchNorm / LayerNorm** | normalizações de ativações (por lote / por vetor) que estabilizam o treino |
| **AMP** | *automatic mixed precision*: cálculo em `float16` onde seguro |
| **OneCycle** | agenda de taxa de aprendizado que sobe e depois desce suavemente |
| **Hash** | "impressão digital" de um conteúdo; hashes iguais ⇒ conteúdos iguais |
| **Nowcasting / forecasting** | estimar o estado atual / prever o futuro |
| **Baseline** | previsor simples usado como régua; aqui, os que não usam imagens (zero, média, histórico de Vmax) |
| **Bootstrap por ciclone** | reamostragem de ciclones (e não de janelas) com reposição, para estimar o IC e o p-valor de uma diferença entre modelos |
| **Pseudorreplicação** | tratar medidas dependentes (p. ex. seeds avaliando as mesmas janelas) como independentes, inflando a significância |
| **POD / FAR / CSI** | probabilidade de detecção / razão de alarmes falsos / índice crítico de sucesso (acertos ÷ (acertos + perdas + falsos alarmes)) |
| **AUC-ROC / AP** | área sob a curva ROC / precisão média (área sob precisão × revocação); medem como o escore ordena os eventos, sem fixar limiar |

## 17. Referências

```text
ARNOLD, V. I. On functions of three variables. Doklady Akademii Nauk SSSR, v. 114, p. 679–681, 1957.

BA, J. L.; KIROS, J. R.; HINTON, G. E. Layer normalization. arXiv:1607.06450, 2016.

CHEN, B.; CHEN, B.-F.; LIN, H.-T. Rotation-blended CNNs on a new open dataset for tropical cyclone
image-to-intensity regression. In: Proceedings of the 24th ACM SIGKDD International Conference on
Knowledge Discovery & Data Mining, 2018. p. 90–99.

CHEN, R. T. Q. et al. Neural ordinary differential equations. Advances in Neural Information Processing
Systems (NeurIPS), v. 31, 2018.

COX, M. G. The numerical evaluation of B-splines. IMA Journal of Applied Mathematics, v. 10, n. 2,
p. 134–149, 1972.

DAVIS, J.; GOADRICH, M. The relationship between precision-recall and ROC curves. ICML, 2006.

DE BOOR, C. On calculating with B-splines. Journal of Approximation Theory, v. 6, n. 1, p. 50–62, 1972.

EFRON, B.; TIBSHIRANI, R. J. An introduction to the bootstrap. Chapman & Hall, 1993.

HOCHREITER, S.; SCHMIDHUBER, J. Long short-term memory. Neural Computation, v. 9, n. 8, p. 1735–1780, 1997.

HOLM, S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics,
v. 6, n. 2, p. 65–70, 1979.

HURLBERT, S. H. Pseudoreplication and the design of ecological field experiments. Ecological
Monographs, v. 54, n. 2, p. 187–211, 1984.

IOFFE, S.; SZEGEDY, C. Batch normalization: accelerating deep network training by reducing internal
covariate shift. ICML, 2015.

KAPLAN, J.; DEMARIA, M. Large-scale characteristics of rapidly intensifying tropical cyclones in the
North Atlantic basin. Weather and Forecasting, v. 18, n. 6, p. 1093–1108, 2003.

KINGMA, D. P.; BA, J. Adam: a method for stochastic optimization. ICLR, 2015.

KOLMOGOROV, A. N. On the representation of continuous functions of several variables by superposition
of continuous functions of one variable and addition. Doklady Akademii Nauk SSSR, v. 114,
p. 953–956, 1957.

LIU, Z. et al. KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756, 2024.

LOSHCHILOV, I.; HUTTER, F. Decoupled weight decay regularization (AdamW). ICLR, 2019.

SELVARAJU, R. R. et al. Grad-CAM: Visual explanations from deep networks via gradient-based
localization. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.
p. 618–626.

SMITH, L. N.; TOPIN, N. Super-convergence: very fast training of neural networks using large learning
rates (agenda OneCycle). arXiv:1708.07120, 2017.

WILCOXON, F. Individual comparisons by ranking methods. Biometrics Bulletin, v. 1, n. 6, p. 80–83, 1945.
```

(A referência sobre RNN, ELMAN, 1990, já consta na bibliografia do TCC.)
