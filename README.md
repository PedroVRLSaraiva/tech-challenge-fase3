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

> _TODO: detalhar aqui, após a EDA, quais tabelas/colunas foram efetivamente
> usadas, volume de linhas e período coberto._

## Etapas de modelagem

> _TODO: descrever a pipeline construída — imputação, encoding, prevenção de
> data leakage, integração do pré-processamento ao modelo via
> `sklearn.pipeline.Pipeline`, estratégia de split/validação._

## Escolha do algoritmo

> _TODO: qual(is) algoritmo(s) foram testados e por quê._

## Métricas de avaliação

> _TODO: métricas usadas (ex.: acurácia, F1, ROC-AUC) e resultados obtidos._

## Interpretação dos resultados

> _TODO: Feature Importance / SHAP — quais variáveis mais influenciam a
> predição e o que isso significa no contexto educacional._

## Insights encontrados

> _TODO: principais achados da EDA e da modelagem._

## Limitações do projeto

> _TODO: limitações de dados, do modelo, e do escopo._

## Aplicação prática para políticas públicas

> _TODO: quais municípios apresentam maior risco, como o modelo apoiaria
> decisões de gestores públicos._

## Possíveis evoluções futuras

> _TODO._

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

> _TODO: comandos para rodar a pipeline/notebooks depois que existirem._

## Vídeo executivo

> _TODO: link do vídeo executivo (até 5 min)._
