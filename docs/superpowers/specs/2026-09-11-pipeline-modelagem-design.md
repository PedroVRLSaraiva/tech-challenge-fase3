# Design — Pipeline de modelagem (Tech Challenge Fase 3)

Data: 2026-09-11
Status: aprovado para virar plano de implementação

## Contexto

A EDA (`notebooks/01_eda_gold_e_alunos.ipynb`) já travou três premissas que
este design assume como dadas e não reabre:

1. **Granularidade: aluno, híbrida.** Unidade de análise é o aluno
   individual (fonte pública `basedosdados.br_inep_avaliacao_alfabetizacao.alunos`),
   enriquecido com território/meta da camada Gold (`indicador_por_municipio`)
   via `id_municipio`.
2. **Escopo: somente `presenca == '1'`.** Remove a regra determinística
   "ausente ⇒ não alfabetizado" (H5).
3. **Sem fontes externas nesta rodada** (IBGE, Censo Escolar, FUNDEB, PNAD,
   Atlas do Desenvolvimento Humano ficam para uma iteração futura).
4. **`proficiencia` é excluída por vazamento** (H6): reproduz o target com
   ~100% de concordância porque o target é definido a partir dela.

Este design cobre a etapa seguinte: transformar essas premissas em uma
pipeline de Machine Learning reprodutível, dos dados brutos até um modelo
treinado, validado e interpretável.

## Objetivo

Construir uma pipeline de classificação supervisionada (`alfabetizado` /
`não alfabetizado`) que:

- Trate valores faltantes e variáveis categóricas de forma integrada ao
  modelo (não manualmente antes do `fit`);
- Evite vazamento de dado, incluindo um tipo de vazamento específico deste
  projeto: vazamento pelo processo de avaliação (reuso do conjunto de teste
  durante iteração);
- Generalize para um ano não visto (2024), não só para linhas não vistas
  dentro do mesmo ano;
- Seja interpretável (Feature Importance / SHAP), com resultado conectado a
  3 das 5 perguntas de negócio do desafio (as outras 2 são tratadas à parte
  — ver "Fora de escopo").

## Camada de dados: nova tabela Gold materializada

**Decisão:** em vez de repetir o JOIN aluno↔território em cada notebook, um
script em `src/preprocessing` materializa uma tabela nova dentro do dataset
BigQuery `gold_alfabetizacao`:

```
fiapfase2.gold_alfabetizacao.aluno_alfabetizado_enriquecido
```

Regras de construção (fixadas aqui, implementadas uma vez, reprocessáveis):

- Uma linha por aluno-ano, filtrado a `presenca == '1'`.
- Enriquecida com território/meta de `indicador_por_municipio` via
  `id_municipio`, sempre usando o ano **anterior** ao ano do aluno (ex.:
  aluno de 2024 recebe território/meta de 2023) — evita vazar o resultado
  do próprio período.
- `proficiencia` não é incluída na tabela (nem como feature, nem por engano
  em consultas futuras).
- Colunas de saída: identificadores (`id_aluno`, `id_municipio`, `ano`),
  target (`alfabetizado`), categóricas (`rede`, `regiao`, `sigla_uf`),
  numéricas territoriais defasadas (`meta_alfabetizacao_ano_anterior`,
  `gap_meta_resultado_ano_anterior`, `taxa_alfabetizacao_ano_anterior`).

Não altera o pipeline `process_gold` do repositório da Fase 2 (repositório
separado, já entregue) — ver
[[ensinamentos/02-modelagem/01-materializacao-de-camada-gold-versionada]].

## Split: treino / validação / teste

- **Dentro de 2023:** split fixo 70% treino / 15% validação / 15%
  teste-2023, estratificado por `alfabetizado`.
- **2024 inteiro:** segundo teste, out-of-time, nunca usado em treino ou
  tuning.
- Ambos os splits usam amostragem determinística (seed fixa), reprodutível
  entre execuções.

Ver [[ensinamentos/02-modelagem/02-split-temporal-out-of-time-validation]].

**Disciplina de iteração:** toda comparação de modelo/hiperparâmetro durante
o desenvolvimento usa a validação. Teste-2023 e 2024 só são avaliados **uma
vez**, depois que o modelo final já estiver decidido. Ver
[[ensinamentos/02-modelagem/04-disciplina-validacao-vs-teste-no-loop-de-iteracao]].

## Features

**Numéricas** (defasadas em 1 ano, da tabela enriquecida):
`meta_alfabetizacao_ano_anterior`, `gap_meta_resultado_ano_anterior`,
`taxa_alfabetizacao_ano_anterior`.

**Categóricas:** `rede`, `regiao`, `sigla_uf`.

**Explicitamente excluídas:**
- `proficiencia` — vazamento (H6).
- `presenca` — constante (`== '1'`) após o filtro de escopo, sem informação.
- `id_aluno`, `id_escola`, `id_municipio` como identificadores brutos —
  cardinalidade alta demais, não generaliza para escola/aluno/município
  novo.
- `ano` — constante dentro de cada partição de treino (é a própria chave do
  split), sem informação para o modelo.

## Pré-processamento (integrado ao modelo via `sklearn.pipeline.Pipeline`)

- **Numéricas:** `SimpleImputer(strategy="median", add_indicator=True)`.
  Mediana é robusta a outlier; `add_indicator=True` preserva o sinal de
  que a ausência de meta municipal é MAR, não aleatória (achado da EDA).
- **Categóricas:** `OneHotEncoder(handle_unknown="ignore")`. Cardinalidade
  baixa/moderada (`rede` ~5-7, `regiao` 5, `sigla_uf` 27) — adequada para
  one-hot. `handle_unknown="ignore"` evita erro se 2024 tiver uma categoria
  não vista no treino de 2023.
- As duas transformações entram num `ColumnTransformer`, e o
  `ColumnTransformer` entra num `Pipeline` junto com o classificador —
  transformers são ajustados (`fit`) **só** no conjunto de treino, e
  aplicados (`transform`) em validação/teste/2024 sem re-ajustar.

## Modelo e seleção

**Candidatos** (todos com `class_weight="balanced"`):

1. `LogisticRegression` — baseline interpretável (coeficientes).
2. `RandomForestClassifier` — não-linear, `feature_importances_` nativo,
   compatível com SHAP `TreeExplainer`.
3. `HistGradientBoostingClassifier` — gradient boosting nativo do
   scikit-learn, eficiente no volume de dado do projeto (~3,9M linhas),
   sem adicionar dependência nova (XGBoost/LightGBM).

**Critério de seleção:** PR-AUC (`average_precision_score`) na validação —
métrica threshold-independente, apropriada para classe minoritária sob
desbalanceamento moderado. O candidato com melhor PR-AUC na validação é o
único avaliado contra teste-2023 e 2024.

## Avaliação e relato de métricas

- **Modelo-a-modelo (seleção):** PR-AUC na validação.
- **Relato final (README):** para o modelo vencedor, reportar:
  - ROC-AUC e PR-AUC (teste-2023 e 2024, lado a lado).
  - Curva precisão-recall.
  - Tabela com 2-3 limiares nomeados: limiar padrão (0.5), limiar que
    maximiza F1, limiar recall-priorizado (F2 ou recall-alvo ≥ 80%) — sem
    travar um único limiar "oficial", já que a escolha do ponto de operação
    depende de uma restrição de capacidade operacional que este desafio não
    fornece (dado simulado). O README deixa essa escolha explícita como
    decisão do gestor público, não do modelo.
- **Checks de variação temporal (antes de interpretar o gap 2023→2024):**
  comparar `% alfabetizado` e distribuição das features usadas entre 2023 e
  2024 (value_counts / teste de Kolmogorov-Smirnov), para diagnosticar se
  uma eventual queda de métrica é covariate shift ou concept shift. Ver
  [[ensinamentos/02-modelagem/03-covariate-shift-vs-concept-shift]].

## Interpretabilidade

- `feature_importances_` (RandomForest/HistGradientBoosting) ou
  coeficientes (LogisticRegression) do modelo vencedor.
- SHAP values via `TreeExplainer` (se o vencedor for uma árvore/ensemble) ou
  `LinearExplainer` (se for a regressão logística), calculados sobre uma
  **amostra** (alguns milhares de linhas, não o dataset completo) — treino
  e teste usam o dataset completo, só o cálculo de SHAP é amostrado, por
  custo computacional (SHAP escala com número de linhas mesmo na variante
  mais eficiente, TreeSHAP).

## Perguntas de negócio cobertas por este design

- ✅ "Quais fatores mais impactam a alfabetização" — feature importance /
  SHAP do modelo vencedor.
- ✅ "Quais municípios apresentam maior risco educacional" — agregação das
  probabilidades previstas por `id_municipio` (ex.: % de alunos previstos
  em risco por município).
- ✅ "Quais variáveis têm maior influência nas predições" — mesmo output de
  SHAP/feature importance.
- ⚠️ "Como prever municípios em risco de não atingir metas futuras" —
  parcialmente coberta: a agregação de predições por município serve de
  proxy, mas uma resposta completa (comparar predição agregada com a meta
  municipal) é um passo de análise em cima do modelo já treinado, não uma
  peça nova de pipeline.

## Fora de escopo desta spec (decomposição deliberada)

- **"Quais regiões apresentam padrões semelhantes"** — é uma pergunta de
  clusterização (não-supervisionada), uma tarefa de natureza diferente da
  pipeline de classificação. A EDA já deu um primeiro passo (teste de
  Kruskal-Wallis confirmando padrão regional), mas uma análise de
  clusterização completa merece seu próprio ciclo de brainstorming — não
  faz parte deste design para não misturar dois tipos de modelagem numa
  única spec/plano de implementação.
- **Fontes externas** (IBGE, Censo Escolar, FUNDEB, PNAD, Atlas do
  Desenvolvimento Humano) — adiadas por decisão já registrada na EDA.
- **Definição de um limiar de decisão único "oficial"** — deliberadamente
  não decidido aqui (ver seção de avaliação); depende de uma restrição de
  capacidade operacional externa ao escopo do desafio.
- **Automação de retreino / monitoramento em produção** — fora do escopo de
  um Tech Challenge acadêmico.

## Estrutura de código

- `src/preprocessing/` — script de materialização da tabela Gold
  (`aluno_alfabetizado_enriquecido`), `ColumnTransformer` (imputação +
  encoding).
- `src/modeling/` — definição dos 3 candidatos, treino, seleção por
  validação (PR-AUC).
- `src/evaluation/` — métricas, curva precisão-recall, checks de variação
  temporal (covariate/concept shift), agregação de predições por
  município.
- `src/visualization/` — gráficos de suporte (curva PR, importância de
  features, SHAP summary plot, comparação de distribuições 2023 vs. 2024).
- `notebooks/02_preprocessing_e_modelagem.ipynb` — notebook que orquestra
  as funções de `src/` e documenta o raciocínio (mesmo formato
  Motivo→Resultado→Decisão do notebook de EDA).
