# Documentação técnica das decisões analíticas

**Projeto:** Predição e Inteligência Analítica para Alfabetização no Brasil
— Tech Challenge Fase 3 (FIAP pós-graduação)
**Escopo deste documento:** registrar, para um leitor técnico, **cada
decisão de modelagem tomada, as alternativas consideradas e por que foram
descartadas** — o "porquê" por trás do pipeline, não só o resultado final.
O [`README.md`](../README.md) do projeto cobre a leitura executiva
(contexto, resultados, aplicação prática); este documento cobre a leitura
de engenharia/metodologia.

**Fontes primárias desta documentação:**
[`docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md`](../docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md)
(design original), os notebooks executados
(`notebooks/01_eda_gold_e_alunos.ipynb`,
`notebooks/02_preprocessing_e_modelagem.ipynb`, cada célula markdown no
formato **Motivo → Resultado → Decisão**), e o código-fonte testado em
`src/` (37 testes, `pytest`).

---

## 1. Escopo e premissas herdadas da EDA

Antes de qualquer decisão de pipeline, a EDA (`notebooks/01_eda_gold_e_alunos.ipynb`)
resolveu uma pergunta anterior: **em que granularidade modelar** —
aluno individual ou município? A tabela abaixo resume a comparação real
feita na EDA (Hipóteses H1-H7):

| critério | nível **aluno** (híbrido) | nível **município** (Gold pura) |
|---|---|---|
| Fidelidade ao enunciado ("prever se um **aluno** será alfabetizado") | Alta | Baixa |
| Volume de dados | ~3,87M linhas | ~24k linhas (município×ano×rede) |
| Balanceamento do target | ~41%/59% (H3, só presentes) | 84,5%/15,5% (`atingiu_meta`, H1) |
| Direto da camada Gold | Não (Silver/fonte pública) | Sim |
| Vazamento explícito conhecido | `proficiencia` (H6, contornável excluindo a coluna) | Menor, mas existe |

**Decisão:** granularidade de **aluno**, enriquecido com território/meta da
Gold via `id_municipio` (join validado em H7: 5.083 dos 5.084 municípios da
amostra de aluno existem na Gold). Fidelidade ao enunciado e balanceamento
do target pesaram mais do que a conformidade estrita à camada Gold. Duas
premissas adicionais, também travadas na EDA e nunca reabertas depois:

- **Escopo restrito a `presenca == '1'`** (Hipótese H5): remove a regra
  determinística "ausente ⇒ não alfabetizado" (13,2% dos registros tinham o
  target decidido pela logística da prova, não por aprendizado).
- **Sem fontes externas nesta rodada** (IBGE, Censo Escolar, FUNDEB, PNAD,
  Atlas do Desenvolvimento Humano) — decisão de escopo para validar a
  pipeline ponta a ponta antes de somar complexidade de integração.

---

## 2. Materialização de uma camada Gold enriquecida

**Problema:** o modelo trabalha em nível de aluno, mas os microdados de
aluno não têm nenhuma variável territorial — só a Gold (`indicador_por_municipio`)
tem `taxa_alfabetizacao`, `gap_meta_resultado` e `meta_alfabetizacao_2024`
por município/ano. Repetir esse join em cada notebook duplicaria uma regra
sutil de prevenção de vazamento (ver seção 3).

**Decisão:** `src/preprocessing/gold_materialization.py` materializa, uma
única vez, `fiapfase2.gold_alfabetizacao.aluno_alfabetizado_enriquecido` —
uma linha por aluno-ano, com as variáveis territoriais do **ano anterior**
ao ano do aluno. Regras fixadas no código, não deixadas para o notebook
decidir ad-hoc:

- **Rede de referência para o território: `Municipal`.** `indicador_por_municipio`
  tem várias linhas por município/ano (uma por rede) — sem filtrar por uma
  rede fixa, o join geraria fanout (múltiplas linhas de território por
  aluno). `Municipal` foi escolhida, não a mesma rede usada na EDA para o
  teste de Kruskal-Wallis (`Pública (Estadual e Municipal)`), porque é a
  única rede com `gap_meta_resultado` e `meta_alfabetizacao_2024`
  populados na Gold real — as outras redes tinham essas colunas 100% nulas,
  o que o `SimpleImputer` descartaria silenciosamente do pipeline (bug real
  encontrado e corrigido durante o desenvolvimento — commit `f815d56`).
- **Território do ano anterior, com fallback documentado.** Só existem 2
  anos de Gold (2023, 2024). Para o cohort de treino de 2023, "ano
  anterior" (2022) não existe — a materialização cai para o **mesmo ano**
  do aluno como fallback. Isso introduz um vazamento fraco e diluído (ver
  seção 3), aceito conscientemente por não haver outra opção com o dado
  disponível.

---

## 3. Prevenção de vazamento de dados

Vazamento foi tratado em três frentes distintas, cada uma com um mecanismo
de prevenção diferente:

1. **Vazamento por definição do target (`proficiencia`).** A regra
   "`proficiencia >= 743` pontos" reproduz `alfabetizado` com 100,0000% de
   concordância numa amostra de checagem (Hipótese H6) — não é uma feature
   fraca, é a própria definição do target. **Decisão:** `proficiencia`
   nunca entra na lista de features candidatas, em nenhuma consulta SQL
   downstream (nem por engano — a tabela materializada nem inclui a
   coluna).
2. **Vazamento temporal (território do próprio período).** Usar o
   indicador do **mesmo ano** do aluno como feature vazaria o resultado do
   próprio período sendo avaliado. **Decisão:** todo enriquecimento usa o
   território do ano **anterior**, exceto no fallback do item seguinte.
3. **Vazamento fraco e diluído (fallback de ano em 2023).** Quando o ano
   anterior não existe (só o cohort de 2023 é afetado), a materialização
   usa o território do mesmo ano como fallback. Isso é qualitativamente
   diferente do vazamento de `proficiencia`: o valor que "vaza" é um
   agregado municipal compartilhado por centenas/milhares de alunos do
   mesmo município naquele ano, não uma cópia disfarçada do rótulo
   individual. Ainda assim é um viés otimista real, isolado ao cohort
   2023 — o teste out-of-time de 2024 não sofre disso, porque para ele o
   ano anterior (2023) está genuinamente disponível. Alternativas
   descartadas: inverter treino/teste (inverteria a narrativa de negócio,
   de "prever o futuro" para "prever o passado"); remover as features
   territoriais defasadas (reduziria demais o escopo de interpretabilidade
   do projeto).
4. **Vazamento pelo processo de avaliação.** Um quarto tipo, menos óbvio:
   escolher o modelo "espiando" o conjunto de teste durante a iteração.
   Prevenido pela disciplina de split (seção 4) e pela decisão de nunca
   tocar `teste_mesmo_ano`/`teste_futuro` antes do modelo final estar
   decidido.

---

## 4. Split treino / validação / teste

**Decisão:** split **temporal**, não aleatório. Dentro de 2023: 70%
treino / 15% validação / 15% teste-mesmo-ano, estratificado por
`em_risco`. O ano de **2024 inteiro** é reservado como segundo teste,
*out-of-time*, nunca usado em treino ou seleção de modelo.

| conjunto | linhas | `em_risco` |
|---|---|---|
| Treino | 1.052.140 | 41,62% |
| Validação | 225.459 | 41,62% |
| Teste-2023 (mesmo ano) | 225.459 | 41,62% |
| Teste-2024 (out-of-time) | 1.852.788 | 40,25% |

**Por que não um split aleatório comum (embaralhar tudo, separar 80/20):**
não responderia a pergunta real de negócio — "o modelo treinado com dados
de um ano consegue generalizar para o ano seguinte, ou só decorou padrões
específicos de 2023?" — porque deixaria linhas de 2024 vazarem para o
treino. `ORDER BY id_aluno, ano` explícito na consulta ao BigQuery garante
que o split seja reprodutível: sem ordenação fixa, `SELECT *` retorna
linhas em ordem arbitrária a cada execução, o que deslocaria o resultado de
`train_test_split`/`StratifiedKFold` mesmo com `random_state` fixo.

---

## 5. Pré-processamento integrado ao modelo

**Decisão:** `SimpleImputer(strategy="median", add_indicator=True)` para as
3 numéricas (`taxa_alfabetizacao_ano_anterior`, `gap_meta_resultado_ano_anterior`,
`meta_alfabetizacao_ano_anterior`) e `OneHotEncoder(handle_unknown="ignore")`
para as 3 categóricas (`rede`, `regiao`, `sigla_uf`), combinados num
`ColumnTransformer` e integrados a um `sklearn.pipeline.Pipeline` junto com
o classificador — nunca aplicados manualmente antes do `fit`.

- **Mediana, não média:** robusta a outlier, sem assumir distribuição
  normal (a EDA já mostrou que `taxa_alfabetizacao` não é normal).
- **`add_indicator=True`:** preserva o sinal de que a ausência de meta
  municipal é **MAR** (Missing At Random condicionado a outra variável —
  nem toda meta municipal está definida na fonte), não MCAR — descartar
  essa informação jogaria fora um padrão real encontrado na EDA.
- **`handle_unknown="ignore"`:** evita erro se 2024 tiver uma categoria de
  `sigla_uf`/`rede` não vista no treino de 2023 (não é hipotético — `SP`
  tinha ~0% de cobertura em 2023 e passou a ~21,6% em 2024, ver seção 8).
- **Por que dentro do `Pipeline`, não manual:** o imputador e o encoder são
  **ajustados (`fit`) só no treino** (ou no pool de desenvolvimento, na
  seleção por validação cruzada — seção 6) e depois só **aplicados**
  (`transform`) em validação/teste — do contrário, a mediana ou as
  categorias usadas para preencher valores faltantes carregariam
  informação estatística do próprio conjunto que se está avaliando de
  forma independente.

---

## 6. Modelo e seleção: de holdout único para validação cruzada

**Candidatos** (todos com `class_weight="balanced"`, compensando o
desbalanceamento moderado ~41%/59%): `LogisticRegression` (baseline
interpretável), `RandomForestClassifier`, `HistGradientBoostingClassifier`
(gradient boosting nativo do scikit-learn, sem dependência nova como
XGBoost/LightGBM, eficiente no volume do projeto).

**Decisão de critério:** PR-AUC (`average_precision_score`) em vez de
ROC-AUC — métrica threshold-independente e mais informativa quando a classe
de interesse (~41%) não é maioria nem minoria extrema; ROC-AUC seria
otimista demais por também premiar bom desempenho na classe majoritária,
que aqui não é o que importa.

**Evolução do critério de seleção — holdout único → validação cruzada.** A
versão inicial da pipeline selecionava o vencedor com um único fit no
treino e uma única medição de PR-AUC na validação (15% de 2023). Isso foi
revisado para **validação cruzada estratificada (`StratifiedKFold`, 5
folds)** sobre um pool de desenvolvimento maior (treino+validação
fundidos, 1.277.599 linhas — ~85% de 2023), porque um único split tem
variância: o resultado depende de quais linhas caíram em validação por
acaso, um risco concreto aqui já que a diferença entre candidatos é pequena.

| modelo | PR-AUC médio (5-fold CV) | desvio entre folds |
|---|---|---|
| **hist_gradient_boosting** | **0,5969** | 0,0007 |
| regressão logística | 0,5954 | 0,0006 |
| random forest | 0,5914 | 0,0004 |

`hist_gradient_boosting` venceu, mas por margem pequena — a diferença entre
o 1º e o 3º colocado (~0,006) é menor que o desvio-padrão entre folds de
cada candidato individualmente. O baixo desvio entre folds mostra que o
resultado é estável, não sorte de um split específico. O vencedor é então
refitado no pool de desenvolvimento inteiro, virando o modelo de produção.

**Por que não regularização/tuning de hiperparâmetros mais agressivo:**
fora de escopo deste ciclo — os três candidatos usam hiperparâmetros
default (exceto `class_weight`); a corrida acirrada entre eles (margem
~0,006) sugere que o teto de desempenho aqui é mais limitado pelas
features disponíveis (3 numéricas + 3 categóricas, sem fonte
socioeconômica externa) do que pela escolha fina de hiperparâmetro —
ver seção 11 (Limitações).

---

## 7. Limiares de decisão: de "escolhido na validação" para "out-of-fold"

Fundir treino+validação num único pool de desenvolvimento para o k-fold
(seção 6) criou um problema novo, não presente na versão de holdout único:
a etapa de calibrar os limiares de decisão (`tabela_limiares`) rodava o
modelo já treinado sobre um conjunto que ele não tinha visto — antes, esse
conjunto era a validação (nunca usada no `fit`). Como o modelo final agora
é refitado no pool de desenvolvimento **inteiro**, não sobra nenhum pedaço
dele que o modelo não tenha visto: calcular `predict_proba` diretamente
sobre o pool usaria dado de treino para calibrar o limiar — um vazamento
sutil (limiar otimista, calibrado sobre previsões que o modelo já
"decorou").

**Decisão:** `cross_val_predict` gera **probabilidades out-of-fold (OOF)**
— para cada linha do pool de desenvolvimento, uma probabilidade prevista
por um modelo treinado nas outras 4 fatias, nunca na fatia que contém
aquela linha. Os 3 cenários de limiar nomeados (padrão, F1-ótimo,
recall-prioritário ≥80%) são localizados sobre essas probabilidades OOF, e
só então aplicados — uma única vez, já fixados — ao teste-2024:

| cenário | limiar (OOF no dev) | precisão em 2024 | recall em 2024 |
|---|---|---|---|
| padrão (0,5) | 0,500 | 0,484 | 0,655 |
| otimizado para F1 | 0,393 | 0,445 | 0,868 |
| recall-prioritário (≥80%) | 0,433 | 0,460 | 0,799 |

**Por que não travar um único limiar "oficial":** a escolha do ponto de
operação depende de uma restrição de capacidade operacional real (quantos
casos um programa de reforço escolar consegue atender) que o desafio não
fornece como dado — travar um número aqui seria inventar uma restrição
inexistente. Essa é uma decisão de escopo consciente, documentada também
no README ("Limitações do projeto").

---

## 8. Diagnóstico do gap de generalização (2023 → 2024)

| conjunto | ROC-AUC | PR-AUC |
|---|---|---|
| Teste-2023 (mesmo ano) | 0,6879 | 0,5964 |
| Teste-2024 (out-of-time) | 0,6384 | 0,5273 |

Antes de aceitar essa queda (~0,05 ROC-AUC, ~0,07 PR-AUC) como "o modelo
generaliza mal", `src/evaluation/variacao_temporal.py` testa a hipótese de
**covariate shift** (distribuição de entrada muda, relação feature→target
continua igual) contra **concept shift** (a própria relação muda) —
diagnósticos diferentes exigem ações corretivas diferentes.

- Teste de Kolmogorov-Smirnov: as 3 features numéricas mudaram de
  distribuição de forma estatisticamente significativa entre 2023 e 2024
  (p ≈ 0,0 nas três).
- Nas categóricas, o achado mais revelador: `sigla_uf = SP` tinha **~0% de
  cobertura em 2023** e passa a **21,6% em 2024** na mesma fonte pública —
  desloca mecanicamente `regiao = Sudeste` (24,1% → 40,8%).

**Decisão:** classificado como **covariate shift** — mudança na composição
geográfica da amostra disponível na fonte pública, não uma mudança real no
comportamento educacional dos alunos — com um co-fator adicional: o
desaparecimento do vazamento fraco do fallback de ano (seção 3, item 3),
presente só no cohort de 2023, que infla artificialmente o desempenho
medido nesse ano. As duas explicações não são mutuamente exclusivas, e com
só 2 anos de dado não é possível isolar quanto cada uma contribui para o
gap observado.

---

## 9. Calibração: probabilidade prevista vs. frequência real

A curva de calibração (10 faixas de probabilidade, `reports/imagens/calibracao_2024.png`)
compara a probabilidade média prevista com a taxa real observada em cada
faixa, em 2024. O modelo fica **sistematicamente abaixo da diagonal**
(previsto 50% → real ~38%; previsto 75% → real ~62%), com o desvio
crescendo nas faixas mais altas.

**Decisão:** o modelo está **descalibrado por superestimação** em 2024,
não apenas "um pouco menos preciso" — consistente com a mesma causa do
gap da seção 8 (o modelo aprendeu, em 2023, um nível de risco "de base"
mais alto do que 2024 de fato apresenta, parcialmente inflado pelo
vazamento do fallback). **Implicação prática documentada:** as
probabilidades previstas servem para **ranquear** risco relativo entre
municípios/alunos, mas não devem ser lidas literalmente como "chance real
de acontecer" nesse ano — por isso `comparar_risco_real_previsto_por_municipio`
(seção 10) sempre reporta risco previsto **e** real lado a lado, nunca só
a previsão isolada.

---

## 10. Interpretabilidade e agregação por município

Feature importance nativa (do classificador vencedor) complementada por
SHAP (`TreeExplainer`, já que o vencedor é uma árvore/ensemble — ver
`src/modeling/interpretabilidade.py`, que normaliza o formato de saída
diferente entre `RandomForestClassifier` — array 3D — e
`HistGradientBoostingClassifier` — já 2D):

| feature | importância |
|---|---|
| `taxa_alfabetizacao_ano_anterior` | 0,559 |
| `gap_meta_resultado_ano_anterior` | 0,036 |
| `rede_Estadual` | 0,033 |
| `rede_Municipal` | 0,016 |
| `meta_alfabetizacao_ano_anterior` | 0,014 |

Uma única variável — o histórico do próprio município — responde por mais
da metade do poder preditivo. **Ressalva documentada:** essa feature é
excelente para *prever* (o passado se repete estatisticamente), mas não é
uma alavanca que um gestor público consiga acionar — o modelo responde bem
a uma pergunta de **triagem** ("onde o risco está concentrado?"), não a
uma pergunta de **política** ("o que fazer a respeito?"). Ver seção 11.

`src/evaluation/risco_por_municipio.py` agrega as probabilidades previstas
por `id_municipio`, produzindo `reports/artefatos/risco_por_municipio_2024.csv` com
risco previsto, risco real e a diferença entre os dois — responde
diretamente a pergunta de negócio "quais municípios apresentam maior risco
educacional", já incorporando a ressalva de calibração da seção 9.

---

## 11. Clusterização de municípios (complemento não-supervisionado)

A pergunta "quais regiões apresentam padrões semelhantes" foi
deliberadamente deixada fora do design de modelagem original (seção "Fora
de escopo" do spec) por ser uma tarefa de natureza diferente
(não-supervisionada) — resolvida depois, num ciclo próprio, em
`src/evaluation/clusterizacao_regional.py`.

**Decisão de granularidade:** clusterizar **municípios** (5.232, com
indicadores completos em 2024), não as 5 regiões oficiais diretamente —
5 pontos seriam poucos para qualquer método de clustering funcionar de
forma robusta. Os clusters resultantes são então cruzados com a região
oficial via `pd.crosstab`.

**Método:** `StandardScaler` (as 3 features têm escalas diferentes) +
`KMeans`, com `k` escolhido por **silhueta** (não pelo método do cotovelo,
subjetivo) testando k=2 a k=10.

| resultado | valor |
|---|---|
| Municípios clusterizados | 5.232 |
| k escolhido | 2 |
| Silhueta do clustering final | 0,385 |
| Faixa de silhueta testada (k=2..10) | 0,348 – 0,385 |

A faixa estreita de silhueta entre k=2 e k=10 é, em si, um resultado
analítico: os indicadores territoriais formam mais um **contínuo** de
desempenho do que grupos discretos bem separados. Os 2 clusters
encontrados dividem o território em **alto desempenho** (taxa média 77,9%,
gap médio +8,6) e **baixo desempenho** (taxa média 45,9%, gap médio -7,8).
Cruzando com região: nenhuma região é homogênea — Centro-Oeste (74,4%),
Sudeste (70,8%) e Sul (59,3%) concentram maioria de alto desempenho; Norte
(77,6%) e Nordeste (62,3%), maioria de baixo desempenho — mas todas têm
municípios "fora do padrão" da própria região.

![Clusters de municípios (cor) vs. região oficial (marcador)](imagens/clusters_municipios_2024.png)

---

## 12. Mapeamento às perguntas de negócio do desafio

| pergunta de negócio | mecanismo | seção |
|---|---|---|
| Quais fatores mais impactam a alfabetização | Feature importance / SHAP | 10 |
| Quais municípios apresentam maior risco educacional | `comparar_risco_real_previsto_por_municipio` | 10 |
| Quais regiões apresentam padrões semelhantes | Clusterização k-means + crosstab regional | 11 |
| Como prever municípios em risco de não atingir metas futuras | Teste out-of-time (2024), risco agregado por município | 4, 10 |
| Quais variáveis têm maior influência nas predições | Mesmo output de SHAP/feature importance | 10 |

---

## 13. Limitações metodológicas (resumo técnico)

Detalhamento completo no README ("Limitações do projeto"); resumo das que
têm implicação metodológica direta:

- **Discriminação fraca-a-moderada** (ROC-AUC 0,64-0,69): abaixo do
  patamar geralmente aceitável para decisão individual de alto risco — uso
  recomendado é priorização relativa (ranking), não veredito individual.
- **Só 2 anos de Gold (2023/2024):** limita a validação temporal a uma
  única fronteira; não é possível confirmar se o gap de generalização é
  padrão recorrente ou específico dessa transição.
- **Sem fonte externa nesta rodada:** a feature dominante
  (`taxa_alfabetizacao_ano_anterior`) é preditiva, mas não acionável —
  Censo Escolar/FUNDEB trariam alavancas de política real (professores,
  investimento), mas exigiriam nova integração de dado.
- **Clusterização usa só 3 indicadores territoriais**, sem variável
  socioeconômica — silhueta moderada (0,385) reflete essa limitação de
  features, não necessariamente uma limitação do método.
- **Vazamento fraco do fallback de ano** (seção 3, item 3): viés otimista
  real, isolado ao cohort de 2023, sem forma de quantificar sua magnitude
  exata com só 2 anos de dado disponíveis.

---

## 14. Rastreabilidade

- **Design original:** [`docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md`](../docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md)
- **Plano de implementação:** [`docs/superpowers/plans/2026-09-11-pipeline-modelagem.md`](../docs/superpowers/plans/2026-09-11-pipeline-modelagem.md)
- **Notebooks executados** (Motivo→Resultado→Decisão célula a célula):
  [`notebooks/01_eda_gold_e_alunos.ipynb`](../notebooks/01_eda_gold_e_alunos.ipynb),
  [`notebooks/02_preprocessing_e_modelagem.ipynb`](../notebooks/02_preprocessing_e_modelagem.ipynb)
- **Código-fonte testado:** `src/{preprocessing,modeling,evaluation,visualization}/`,
  37 testes em `tests/` (`pytest`)
- **Leitura executiva:** [`README.md`](../README.md)
