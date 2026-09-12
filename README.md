# Predição e Inteligência Analítica para Alfabetização no Brasil

Tech Challenge — Fase 3 (FIAP). Modelo supervisionado para prever se um aluno
será considerado alfabetizado ou não, a partir de variáveis educacionais,
territoriais e socioeconômicas da camada Gold construída na Fase 2.

## Contexto do problema

A alfabetização infantil é um dos principais indicadores do desenvolvimento
educacional e social do país. O **Compromisso Nacional Criança Alfabetizada**
define como alfabetizada a criança que atinge 743 pontos na escala de
proficiência do Saeb até o final do 2º ano do ensino fundamental, com meta
nacional de universalização até 2030. Conhecer os dados atuais não basta:
gestores públicos precisam antecipar riscos, identificar regiões vulneráveis
e entender quais fatores mais pesam nos indicadores educacionais.

## Objetivo analítico

Desenvolver um modelo supervisionado de classificação (alfabetizado /
não alfabetizado) e usá-lo, junto de análise exploratória e técnicas de
interpretabilidade (Feature Importance / SHAP), para responder perguntas de
negócio relevantes para políticas públicas — não apenas maximizar métricas.

## Descrição da base utilizada

Base construída na camada **Gold** do pipeline da Fase 3 (ver
[`FIAP - Tech Challenge - Fase 2 copy`](../FIAP%20-%20Tech%20Challenge%20-%20Fase%202%20copy)),
dataset BigQuery `gold_alfabetizacao`, com as tabelas:

- `indicador_por_municipio` — Indicador Criança Alfabetizada por município;
- `comparacao_meta_resultado` — metas nacionais/estaduais/municipais vs. resultado real;
- `evolucao_temporal` — evolução temporal do indicador.

Poderá ser enriquecida com fontes externas: IBGE, Censo Escolar, FUNDEB,
PNAD, Atlas do Desenvolvimento Humano, Cadastro Único.

A tabela efetivamente usada para treinar o modelo não é nenhuma das três
acima diretamente — é uma tabela nova, materializada especificamente para
esta fase (ver seção seguinte): `fiapfase2.gold_alfabetizacao.aluno_alfabetizado_enriquecido`,
com **3.355.846 linhas** (uma linha por aluno presente na avaliação, por
ano), cobrindo o período **2023-2024** (os dois únicos anos disponíveis na
Gold no momento desta análise). Ela une os microdados públicos de aluno
(fonte `basedosdados.br_inep_avaliacao_alfabetizacao.alunos`) com o
território/meta do município vindo de `indicador_por_municipio`.

Nenhuma fonte externa (IBGE, Censo Escolar, FUNDEB, PNAD, Atlas do
Desenvolvimento Humano) foi incorporada nesta rodada — decisão registrada
na EDA e mantida deliberadamente fora de escopo (ver "Possíveis evoluções
futuras").

## Etapas de modelagem

O detalhamento completo das decisões abaixo — com as alternativas
consideradas e descartadas — está no documento de design:
[`docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md`](docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md).
Resumo do que foi construído:

1. **Materialização de uma tabela Gold enriquecida.** Em vez de repetir o
   JOIN aluno↔território em cada notebook, um script em `src/preprocessing`
   materializa `aluno_alfabetizado_enriquecido` uma única vez no BigQuery:
   uma linha por aluno-ano (só alunos presentes na avaliação), enriquecida
   com o território/meta do município do **ano anterior** ao ano do aluno
   (ex.: aluno de 2024 recebe indicadores de 2023) — isso evita que o
   modelo veja, disfarçado de "contexto territorial", um resultado que só
   existiria depois do fato observado. Quando o ano anterior não existe na
   Gold (caso do cohort de treino de 2023, já que só há 2023/2024
   disponíveis), a lógica cai para o indicador do **mesmo** ano como
   *fallback* — uma concessão a uma limitação real de dado, detalhada em
   "Limitações do projeto".
2. **Split temporal, não aleatório.** Dentro de 2023, os dados são
   divididos 70% treino / 15% validação / 15% teste, estratificados por
   `alfabetizado`. O ano de **2024 inteiro** é reservado como um segundo
   teste, *out-of-time* — nunca usado em treino ou ajuste de
   hiperparâmetros. A motivação é simular a pergunta real de negócio: o
   modelo treinado com dados de um ano consegue generalizar para o ano
   seguinte, ou só decorou padrões específicos de 2023? Um split aleatório
   comum (embaralhar tudo e separar 80/20) não responderia essa pergunta,
   porque deixaria linhas de 2024 vazarem para o treino.
3. **Pré-processamento integrado ao modelo**, via
   `sklearn.pipeline.Pipeline` + `ColumnTransformer`: as features numéricas
   (`meta_alfabetizacao_ano_anterior`, `gap_meta_resultado_ano_anterior`,
   `taxa_alfabetizacao_ano_anterior`) passam por
   `SimpleImputer(strategy="median", add_indicator=True)`, e as
   categóricas (`rede`, `regiao`, `sigla_uf`) por
   `OneHotEncoder(handle_unknown="ignore")`. Os dois pontos importam para
   prevenção de vazamento: por estarem dentro do `Pipeline`, o imputador e
   o encoder são **ajustados (`fit`) somente no treino** e depois aplicados
   (sem reajustar) em validação, teste-2023 e 2024 — do contrário, a
   mediana ou as categorias usadas para preencher valores faltantes
   carregariam informação estatística do próprio conjunto que se está
   tentando avaliar de forma independente.
4. **Três modelos candidatos** (todos com `class_weight="balanced"` para
   compensar o desbalanceamento moderado da classe "em risco"):
   `LogisticRegression` (baseline interpretável), `RandomForestClassifier`
   e `HistGradientBoostingClassifier`. Os três são comparados **apenas na
   validação**, nunca no teste, para evitar um segundo tipo de vazamento —
   vazamento pelo próprio processo de avaliação (escolher o modelo
   "espiando" o teste).

## Escolha do algoritmo

O critério de seleção foi o **PR-AUC** (`average_precision_score`) medido
na validação — uma métrica que não depende de escolher um limiar de
decisão de antemão e que é mais informativa que a acurácia quando a classe
de interesse (aluno em risco de não alfabetização) não é maioria nem
minoria extrema (aqui, ~41% da base). Resultado da corrida:

| modelo | PR-AUC (validação) |
|---|---|
| **hist_gradient_boosting** | **0,5963** |
| regressão logística | 0,5947 |
| random forest | 0,5912 |

O `HistGradientBoostingClassifier` venceu, mas por margem pequena — a
diferença entre o 1º e o 3º colocado é de apenas ~0,005 em PR-AUC. Vale
registrar que essas três últimas casas decimais oscilam ligeiramente entre
execuções (a consulta ao BigQuery não fixa a ordem das linhas retornadas,
o que desloca minimamente o `train_test_split` mesmo com seed fixa), mas o
vencedor e a conclusão — corrida acirrada entre os três candidatos, sem um
modelo claramente dominante — se mantiveram estáveis em todas as
execuções observadas.

## Métricas de avaliação

O modelo vencedor (`hist_gradient_boosting`) foi avaliado **uma única vez**
contra dois conjuntos de teste que nunca participaram de treino ou seleção
de modelo: teste-2023 (linhas do mesmo ano de treino, nunca vistas) e o ano
de 2024 inteiro (*out-of-time*, um ano que o modelo nunca viu):

| conjunto | ROC-AUC | PR-AUC |
|---|---|---|
| Teste-2023 (mesmo ano) | 0,6899 | 0,5991 |
| 2024 (out-of-time) | 0,6386 | 0,5275 |

Há uma queda real de desempenho de 2023 para 2024 (~0,05 em ROC-AUC, ~0,07
em PR-AUC) — investigada na seção "Insights encontrados" abaixo, onde
mostramos que a causa é uma mudança na composição dos dados de entrada, não
uma falha do modelo em si.

Como a decisão de classificar um aluno como "em risco" depende de um
limiar sobre a probabilidade prevista, e a escolha desse limiar é uma
decisão de política pública (quanto o gestor está disposto a errar para o
lado de "alarme falso" vs. "caso perdido"), reportamos três cenários sobre
o conjunto de 2024, em vez de travar um único limiar "oficial":

| cenário | limiar | precisão | recall |
|---|---|---|---|
| padrão (0,5) | 0,500 | 0,489 | 0,638 |
| otimizado para F1 | 0,361 | 0,439 | 0,912 |
| recall-prioritário (≥80%) | 0,430 | 0,460 | 0,800 |

Ver a leitura de negócio dessa tabela em "Aplicação prática para políticas
públicas".

## Interpretação dos resultados

Importância nativa de features do modelo vencedor (complementada por SHAP
quando a importância nativa não está disponível para todas as colunas
transformadas):

| feature | importância |
|---|---|
| `taxa_alfabetizacao_ano_anterior` | 0,549 |
| `rede_Estadual` | 0,031 |
| `gap_meta_resultado_ano_anterior` | 0,030 |
| `rede_Municipal` | 0,017 |
| `meta_alfabetizacao_ano_anterior` | 0,013 |
| `regiao_Nordeste` | 0,010 |
| `regiao_Sudeste` | 0,007 |

![Importância de features](reports/importancia_features.png)
![Curva precisão-recall](reports/curva_precisao_recall.png)
![Resumo SHAP](reports/shap_summary.png)

**Leitura em termos simples:** uma única variável — a taxa histórica de
alfabetização do próprio município no ano anterior — responde por mais de
metade do poder preditivo do modelo (0,549 de importância, contra 0,031 da
segunda colocada). Isso significa que o fator individual mais forte para
prever se *um aluno específico* será alfabetizado não é uma característica
pessoal daquele aluno, e sim o quão bem o município onde ele estuda já vem
performando recentemente. Em outras palavras: nascer/estudar num município
com histórico consistente de bons resultados é, isoladamente, o preditor
mais forte que este modelo encontrou — o que reforça a leitura de que
alfabetização é, em grande parte, um fenômeno territorial/sistêmico, e não
apenas individual. Rede de ensino (estadual vs. municipal) e a distância
entre meta e resultado do município aparecem como fatores secundários, mas
com peso bem menor.

## Insights encontrados

**1. Existe um gap real de generalização temporal.** O ROC-AUC cai de
0,6899 (teste-2023, mesmo ano de treino) para 0,6386 (2024, ano nunca
visto) — uma queda de ~0,05, com uma queda proporcional maior em PR-AUC
(0,5991 → 0,5275). Isso por si só não diz se o modelo "aprendeu errado" ou
se o mundo mudou entre 2023 e 2024; por isso, antes de aceitar o número,
rodamos um diagnóstico de variação temporal.

**2. O diagnóstico aponta covariate shift, não falha de modelo.** Um teste
de Kolmogorov-Smirnov confirma que as 3 features numéricas mudaram de
distribuição de forma estatisticamente significativa entre 2023 e 2024
(p ≈ 0,0 nas três). O achado mais revelador está nas categóricas:
`sigla_uf = SP` (São Paulo) tinha **~0% de cobertura em 2023** e passa a
representar **21,6% dos alunos em 2024** na mesma fonte pública de dados.
Ou seja: São Paulo praticamente não aparecia na fonte de microdados de
aluno em 2023 e passou a aparecer de forma substancial em 2024 —
mecanicamente, isso desloca toda a distribuição das features (inclusive
`regiao = Sudeste`, que sobe de 24,1% para 40,8% só pelo efeito de SP
entrar). É uma mudança na **composição da amostra de dados disponível**,
não uma mudança real no comportamento educacional dos alunos — o modelo
não "piorou": ele está vendo, em 2024, uma população parcialmente
diferente (mais representativa de SP) daquela em que foi treinado, e essa
mudança de composição explica boa parte da queda de métrica observada.

## Limitações do projeto

- **A Gold só tem 2 anos de dado (2023 e 2024).** Isso limita a validação
  temporal a uma única fronteira (treinar em 2023, testar em 2024) — não é
  possível confirmar se o gap de generalização observado é um padrão
  recorrente ou específico dessa transição de ano até que exista uma
  terceira safra de dado para comparar.
- **Nenhuma fonte externa foi incorporada nesta rodada** (IBGE, Censo
  Escolar, FUNDEB, PNAD, Atlas do Desenvolvimento Humano) — decisão
  deliberada para manter o escopo do desafio tratável; features como
  investimento por aluno (FUNDEB) ou nível socioeconômico (Censo/PNAD)
  provavelmente melhorariam o poder preditivo além do que o território
  histórico sozinho consegue capturar.
- **Não travamos um limiar de decisão único "oficial".** A tabela de
  limiares na seção de métricas apresenta 3 cenários porque a escolha do
  ponto de operação depende de uma restrição de capacidade operacional do
  gestor público (quantos casos ele consegue atender) que este desafio não
  fornece como dado real — travar um número aqui seria inventar uma
  restrição que não existe.
- **A pergunta "quais regiões têm padrões semelhantes" foi tratada apenas
  parcialmente.** A EDA (`notebooks/01_eda_gold_e_alunos.ipynb`) já indicou
  um padrão regional via teste de Kruskal-Wallis, mas isso não substitui
  uma clusterização completa (não-supervisionada) das regiões — que é uma
  tarefa de natureza diferente da pipeline de classificação construída
  aqui e fica como próximo passo (ver "Possíveis evoluções futuras").
- **O fallback de "ano anterior → mesmo ano" introduz um vazamento
  fraco e diluído no cohort de treino de 2023.** Como a Gold só tem
  2023/2024, alunos de 2023 não têm um "ano anterior" real disponível;
  nesse caso, a materialização usa o indicador territorial do **mesmo**
  ano do aluno em vez do ano anterior. Isso é uma concessão consciente, não
  um erro passado despercebido: o vazamento é qualitativamente diferente do
  vazamento por `proficiencia` (que reproduziria o target individual quase
  perfeitamente) porque aqui o valor é um agregado municipal, compartilhado
  por centenas ou milhares de alunos do mesmo município naquele ano — o
  sinal extra que "vaza" é fraco e diluído, não uma cópia disfarçada do
  rótulo. Ainda assim, é um viés otimista real no desempenho medido sobre o
  cohort de treino/validação/teste-2023 (o teste out-of-time de 2024 não
  sofre desse problema, porque para ele o ano anterior — 2023 — está
  genuinamente disponível). Alternativas descartadas: inverter qual ano é
  treino vs. teste (inverteria a narrativa de negócio, de "prever o
  futuro" para "prever o passado"); remover as features territoriais
  defasadas (reduziria demais o escopo de interpretabilidade, que é uma
  das perguntas de negócio centrais do desafio).

## Aplicação prática para políticas públicas

O modelo permite estimar, para cada aluno, uma probabilidade de risco de
não alfabetização — e, agregando essas probabilidades por município, gerar
um **ranking de municípios prioritários** para ação. A tabela completa
está em [`reports/risco_por_municipio_2024.csv`](reports/risco_por_municipio_2024.csv);
no topo do ranking de 2024 aparecem casos como o município de
`id_municipio` 1718501 (risco médio de 0,947 entre 48 alunos), 1718006
(0,942 entre 47 alunos) e 1715705 (0,942 entre 81 alunos) — municípios onde
quase a totalidade dos alunos avaliados está classificada como em risco de
não atingir o patamar de alfabetização esperado. Esse tipo de ranking é
diretamente acionável: em vez de distribuir recursos (formação de
professores, material didático, reforço escolar) de forma uniforme entre
todos os municípios, um gestor estadual ou federal pode priorizar onde o
modelo indica maior concentração de risco.

A tabela de limiares (seção "Métricas de avaliação") existe justamente
para dar ao gestor público — não ao modelo — o controle sobre um trade-off
que é uma decisão de política, não uma decisão técnica:

- **Limiar recall-prioritário (recall ≥ 80%):** captura 8 em cada 10 alunos
  realmente em risco, ao custo de uma precisão menor (46%) — ou seja, entre
  os alunos sinalizados, quase metade não estava de fato em risco. Faz
  sentido quando o custo de "deixar passar" um caso real é alto (ex.: uma
  política de reforço escolar barata e escalável, onde é aceitável incluir
  alguns alunos que não precisariam).
- **Limiar padrão (0,5):** um meio-termo (63,8% de recall, 48,9% de
  precisão).
- **Limiar otimizado para F1:** maximiza o equilíbrio entre as duas
  métricas (91,2% de recall, 43,9% de precisão) — captura quase todos os
  casos de risco, mas com a menor precisão das três opções.

Em outras palavras: quanto mais o gestor prioriza **não deixar nenhum caso
de risco passar despercebido**, mais recursos serão direcionados também a
alunos que, na prática, não precisariam — e vice-versa. Como este desafio
não define a capacidade operacional real de nenhuma rede de ensino
(quantos alunos um programa de reforço consegue atender), a decisão de
qual cenário usar fica deliberadamente com quem vai operacionalizar a
política, não com o modelo.

## Possíveis evoluções futuras

- **Incorporar fontes externas** (IBGE, Censo Escolar, FUNDEB, PNAD, Atlas
  do Desenvolvimento Humano) para capturar fatores socioeconômicos e de
  investimento que o território histórico, sozinho, não representa.
- **Clusterização regional completa** para responder de forma robusta
  "quais regiões apresentam padrões semelhantes" — a EDA já indicou padrão
  regional (Kruskal-Wallis), mas uma análise não-supervisionada dedicada
  (ex.: k-means ou clusterização hierárquica sobre indicadores municipais)
  daria uma resposta mais completa e seria um ciclo de trabalho próprio.
- **Tornar o fallback de ano mais robusto**: hoje a checagem de
  disponibilidade do ano anterior usa o mínimo global da tabela de
  território, não uma checagem por município — funciona corretamente com
  os dados atuais (só 2023/2024, cobertura uniforme), mas deixaria de
  funcionar corretamente se uma carga futura tiver municípios reportando
  com atraso (cobertura não-uniforme entre municípios). O fix correto seria
  checar disponibilidade por `(id_municipio, ano)` em vez do mínimo global
  — registrado aqui para não se perder quando o dado crescer.
- **Monitoramento e retreino periódico**, uma vez que mais anos de Gold
  estejam disponíveis — permitiria confirmar se o gap de generalização
  observado entre 2023→2024 é uma tendência recorrente (mudança de
  cobertura geográfica ano a ano) ou um evento pontual, e automatizar a
  decisão de quando retreinar o modelo.

## Estrutura do repositório

```
.
├── data/                 # dados (brutos/processados) usados no projeto
├── notebooks/            # notebooks de EDA e experimentação
├── src/
│   ├── preprocessing/    # limpeza, imputação, encoding
│   ├── modeling/         # treinamento e seleção de modelo
│   ├── evaluation/       # métricas e validação
│   └── visualization/    # gráficos e visualizações
├── reports/              # relatórios e documentação de decisões
├── images/               # imagens usadas em relatórios/README
├── requirements.txt
└── README.md
```

## Como reproduzir

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Autentique-se no Google Cloud (a pipeline lê e escreve no BigQuery):

```bash
gcloud auth application-default login
```

Depois, execute o notebook de ponta a ponta (materializa a tabela Gold
enriquecida no BigQuery, treina e seleciona o modelo, e regrava os
artefatos em `reports/`):

```bash
venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/02_preprocessing_e_modelagem.ipynb \
  --output 02_preprocessing_e_modelagem.ipynb
```

## Vídeo executivo

> _TODO: link do vídeo executivo (até 5 min)._
