# Pipeline de Modelagem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir, testado e ponta a ponta, o pipeline de ML que transforma a
tabela enriquecida de aluno (nova camada Gold) num modelo de classificação
(`em_risco` = não alfabetizado) treinado, validado, testado contra um ano
futuro (2024) e interpretável via SHAP.

**Architecture:** Cada etapa da spec vira um módulo pequeno e puro em `src/`
(materialização de dado, features/pré-processamento, split, candidatos de
modelo, seleção, métricas, variação temporal, agregação por município,
interpretabilidade, visualização), coberto por testes unitários com dados
sintéticos — sem depender de BigQuery real nos testes. Um notebook final
(`notebooks/02_preprocessing_e_modelagem.ipynb`) importa essas funções e
orquestra a execução real contra o BigQuery, documentando o raciocínio.

**Tech Stack:** Python (venv do projeto), pandas, scikit-learn >= 1.2, scipy,
shap, matplotlib, google-cloud-bigquery, pytest (novo).

**Spec:** `docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md`

## Global Constraints

- Rodar tudo com o Python do venv do projeto: `venv/bin/python3` /
  `venv/bin/python3 -m pytest`.
- `scikit-learn >= 1.2` (necessário para `class_weight` em
  `HistGradientBoostingClassifier` e `sparse_output` em `OneHotEncoder`) —
  fixar em `requirements.txt`.
- Testes unitários nunca chamam BigQuery de verdade — toda função que faz
  I/O externo (BigQuery) recebe o cliente por parâmetro (injeção de
  dependência) e é testada com um objeto fake, nunca com `monkeypatch` do
  SDK real.
- Convenção de rótulo fixada em todo o pipeline: a coluna de target usada
  para treino/avaliação é **`em_risco`** (1 = não alfabetizado / em risco,
  0 = alfabetizado) — o inverso de `alfabetizado`. Isso alinha a classe
  positiva (1) do scikit-learn com a classe de interesse do negócio em
  **toda** métrica (`predict_proba(...)[:, 1]`, `average_precision_score`,
  `precision_recall_curve`) sem precisar lembrar de passar `pos_label` em
  cada chamada.
- Nomes de função/variável em português, consistente com o notebook de EDA
  já existente (`notebooks/01_eda_gold_e_alunos.ipynb`). Docstrings e
  comentários em português.
- Toda tabela nova no BigQuery usa o nome completo
  `fiapfase2.gold_alfabetizacao.aluno_alfabetizado_enriquecido` — não
  inventar nomes alternativos entre tasks.

---

## Task 1: Materialização da camada Gold enriquecida

**Files:**
- Create: `src/preprocessing/gold_materialization.py`
- Test: `tests/preprocessing/test_gold_materialization.py`
- Modify: `requirements.txt` (adicionar `pytest`)
- Create: `conftest.py` (vazio, na raiz do repo — permite `pytest` resolver
  `import src...` a partir da raiz, mesmo padrão do repo da Fase 2)

**Interfaces:**
- Produces:
  - `REGIAO_POR_UF: dict[str, str]`
  - `MAPEAMENTO_REDE: dict[str, str]`
  - `REDE_REFERENCIA_TERRITORIO: str`
  - `TABELA_DESTINO: str`
  - `preparar_territorio_por_ano(indicador_municipio: pd.DataFrame) -> pd.DataFrame`
    — colunas de saída: `id_municipio, ano, sigla_uf, regiao, taxa_alfabetizacao, gap_meta_resultado, meta_alfabetizacao_2024`
  - `enriquecer_alunos(alunos: pd.DataFrame, territorio_por_ano: pd.DataFrame) -> pd.DataFrame`
    — colunas de saída: `id_aluno, id_municipio, ano, alfabetizado, rede, regiao, sigla_uf, taxa_alfabetizacao_ano_anterior, gap_meta_resultado_ano_anterior, meta_alfabetizacao_ano_anterior`
  - `materializar_tabela_enriquecida(client) -> None` — `client` é qualquer
    objeto com `.query(sql).to_dataframe()` e
    `.load_table_from_dataframe(df, tabela, job_config=None)` (duck typing,
    compatível com `google.cloud.bigquery.Client`).

- [ ] **Step 1: Adicionar pytest e criar estrutura de testes**

```bash
echo "pytest" >> requirements.txt
venv/bin/python3 -m pip install pytest
mkdir -p tests/preprocessing tests/modeling tests/evaluation
touch conftest.py
```

- [ ] **Step 2: Escrever os testes das duas funções puras (falhando)**

Criar `tests/preprocessing/test_gold_materialization.py`:

```python
import pandas as pd

from src.preprocessing.gold_materialization import (
    MAPEAMENTO_REDE,
    REGIAO_POR_UF,
    TABELA_DESTINO,
    enriquecer_alunos,
    materializar_tabela_enriquecida,
    preparar_territorio_por_ano,
)


def _indicador_municipio_fake() -> pd.DataFrame:
    return pd.DataFrame({
        "id_municipio": ["3550308", "3550308", "3550308", "1200401"],
        "ano": [2022, 2023, 2023, 2023],
        "rede": [
            "Pública (Estadual e Municipal)",
            "Pública (Estadual e Municipal)",
            "Privada",
            "Pública (Estadual e Municipal)",
        ],
        "sigla_uf": ["SP", "SP", "SP", "AC"],
        "taxa_alfabetizacao": [70.0, 75.0, 90.0, 60.0],
        "gap_meta_resultado": [-5.0, -2.0, 10.0, -8.0],
        "meta_alfabetizacao_2024": [80.0, 80.0, 80.0, None],
    })


def test_preparar_territorio_por_ano_filtra_rede_referencia_e_deriva_regiao():
    resultado = preparar_territorio_por_ano(_indicador_municipio_fake())

    # A linha de rede "Privada" para 3550308/2023 deve ser descartada, e a
    # coluna 'rede' não deve sobreviver no resultado (só serve de filtro aqui).
    assert len(resultado) == 3
    assert "rede" not in resultado.columns

    linha_sp_2023 = resultado.query("id_municipio == '3550308' and ano == 2023").iloc[0]
    assert linha_sp_2023["regiao"] == "Sudeste"
    assert linha_sp_2023["taxa_alfabetizacao"] == 75.0

    linha_ac_2023 = resultado.query("id_municipio == '1200401' and ano == 2023").iloc[0]
    assert linha_ac_2023["regiao"] == "Norte"
    assert pd.isna(linha_ac_2023["meta_alfabetizacao_2024"])


def _alunos_fake() -> pd.DataFrame:
    return pd.DataFrame({
        "id_aluno": ["A1", "A2", "A3", "A4"],
        "ano": [2023, 2023, 2023, 2024],
        "id_municipio": ["3550308", "3550308", "1200401", "3550308"],
        "rede": ["5", "5", "5", "5"],
        "presenca": ["1", "0", "1", "1"],
        "alfabetizado": ["1", "0", "0", "1"],
    })


def test_enriquecer_alunos_filtra_ausentes_mapeia_rede_e_junta_ano_anterior():
    territorio_por_ano = preparar_territorio_por_ano(_indicador_municipio_fake())
    resultado = enriquecer_alunos(_alunos_fake(), territorio_por_ano)

    # A2 tinha presenca == '0' e deve ter sido removida.
    assert set(resultado["id_aluno"]) == {"A1", "A3", "A4"}

    # rede '5' mapeia para o nome completo.
    assert (resultado["rede"] == MAPEAMENTO_REDE["5"]).all()

    # alfabetizado virou inteiro.
    assert resultado["alfabetizado"].dtype.kind in "iu"

    # A1 é aluno de 2023 em 3550308 -> território usa ano anterior (2022).
    linha_a1 = resultado.set_index("id_aluno").loc["A1"]
    assert linha_a1["taxa_alfabetizacao_ano_anterior"] == 70.0

    # A4 é aluno de 2024 em 3550308 -> território usa ano anterior (2023, rede referência).
    linha_a4 = resultado.set_index("id_aluno").loc["A4"]
    assert linha_a4["taxa_alfabetizacao_ano_anterior"] == 75.0

    # proficiencia nunca deve aparecer no resultado (vazamento).
    assert "proficiencia" not in resultado.columns

    # colunas de identificação/target/features presentes.
    colunas_esperadas = {
        "id_aluno", "id_municipio", "ano", "alfabetizado",
        "rede", "regiao", "sigla_uf",
        "taxa_alfabetizacao_ano_anterior", "gap_meta_resultado_ano_anterior",
        "meta_alfabetizacao_ano_anterior",
    }
    assert colunas_esperadas.issubset(set(resultado.columns))


class _FakeQueryJob:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_dataframe(self) -> pd.DataFrame:
        return self._df


class _FakeLoadJob:
    def result(self):
        return None


class _FakeClient:
    def __init__(self, alunos_df: pd.DataFrame, indicador_df: pd.DataFrame):
        self._alunos_df = alunos_df
        self._indicador_df = indicador_df
        self.tabela_carregada = None
        self.dataframe_carregado = None

    def query(self, sql: str) -> _FakeQueryJob:
        if "br_inep_avaliacao_alfabetizacao" in sql:
            return _FakeQueryJob(self._alunos_df)
        return _FakeQueryJob(self._indicador_df)

    def load_table_from_dataframe(self, df, tabela_destino, job_config=None):
        self.tabela_carregada = tabela_destino
        self.dataframe_carregado = df
        return _FakeLoadJob()


def test_materializar_tabela_enriquecida_carrega_no_destino_certo():
    client = _FakeClient(_alunos_fake(), _indicador_municipio_fake())

    materializar_tabela_enriquecida(client)

    assert client.tabela_carregada == TABELA_DESTINO
    assert client.dataframe_carregado is not None
    assert len(client.dataframe_carregado) == 3  # A2 removida por ausência
```

- [ ] **Step 3: Rodar os testes e confirmar que falham (módulo não existe)**

Run: `venv/bin/python3 -m pytest tests/preprocessing/test_gold_materialization.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'src.preprocessing.gold_materialization'`

- [ ] **Step 4: Implementar `src/preprocessing/gold_materialization.py`**

```python
"""Materialização da tabela Gold enriquecida (aluno + território, granularidade
de aluno) — ver docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md."""

import pandas as pd

TABELA_ALUNOS_FONTE = "basedosdados.br_inep_avaliacao_alfabetizacao.alunos"
TABELA_INDICADOR_MUNICIPIO = "fiapfase2.gold_alfabetizacao.indicador_por_municipio"
TABELA_DESTINO = "fiapfase2.gold_alfabetizacao.aluno_alfabetizado_enriquecido"

# Classificação geográfica fixa do IBGE (não é fonte externa nova — é um
# lookup estático usado só para derivar 'regiao' a partir de 'sigla_uf',
# mesmo mapeamento já usado na EDA).
REGIAO_POR_UF = {
    **{uf: "Norte" for uf in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
    **{uf: "Nordeste" for uf in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{uf: "Centro-Oeste" for uf in ["DF", "GO", "MT", "MS"]},
    **{uf: "Sudeste" for uf in ["ES", "MG", "RJ", "SP"]},
    **{uf: "Sul" for uf in ["PR", "RS", "SC"]},
}

MAPEAMENTO_REDE = {
    "0": "Total (Federal, Estadual, Municipal e Privada)",
    "1": "Federal",
    "2": "Estadual",
    "3": "Municipal",
    "4": "Privada",
    "5": "Pública (Estadual e Municipal)",
    "6": "Pública (Federal, Estadual e Municipal)",
}

# Rede usada como referência para enriquecimento territorial: evita fanout
# (indicador_por_municipio tem várias linhas por município/ano, uma por
# rede) e é a mesma rede de referência já usada na EDA (H2) para comparar
# regiões de forma consistente.
REDE_REFERENCIA_TERRITORIO = "Pública (Estadual e Municipal)"

COLUNAS_TERRITORIO = [
    "id_municipio", "ano", "sigla_uf", "regiao",
    "taxa_alfabetizacao", "gap_meta_resultado", "meta_alfabetizacao_2024",
]

COLUNAS_FINAIS_ENRIQUECIDO = [
    "id_aluno", "id_municipio", "ano", "alfabetizado",
    "rede", "regiao", "sigla_uf",
    "taxa_alfabetizacao_ano_anterior", "gap_meta_resultado_ano_anterior",
    "meta_alfabetizacao_ano_anterior",
]


def preparar_territorio_por_ano(indicador_municipio: pd.DataFrame) -> pd.DataFrame:
    """Reduz indicador_por_municipio a 1 linha por (id_municipio, ano), usando
    a rede de referência, e deriva 'regiao' a partir de 'sigla_uf'."""
    territorio = indicador_municipio[
        indicador_municipio["rede"] == REDE_REFERENCIA_TERRITORIO
    ].copy()
    territorio["regiao"] = territorio["sigla_uf"].map(REGIAO_POR_UF)
    territorio["ano"] = territorio["ano"].astype("int64")
    return (
        territorio[COLUNAS_TERRITORIO]
        .drop_duplicates(subset=["id_municipio", "ano"])
        .reset_index(drop=True)
    )


def enriquecer_alunos(alunos: pd.DataFrame, territorio_por_ano: pd.DataFrame) -> pd.DataFrame:
    """Filtra alunos presentes, mapeia rede, e junta com o território do ano
    ANTERIOR ao ano do aluno (evita vazar o resultado do próprio período)."""
    alunos_presentes = alunos[alunos["presenca"] == "1"].copy()
    alunos_presentes["rede"] = alunos_presentes["rede"].map(MAPEAMENTO_REDE)
    alunos_presentes["alfabetizado"] = alunos_presentes["alfabetizado"].astype(int)
    alunos_presentes["ano"] = alunos_presentes["ano"].astype("int64")
    alunos_presentes["ano_anterior"] = alunos_presentes["ano"] - 1

    territorio_renomeado = territorio_por_ano.rename(columns={
        "ano": "ano_anterior",
        "taxa_alfabetizacao": "taxa_alfabetizacao_ano_anterior",
        "gap_meta_resultado": "gap_meta_resultado_ano_anterior",
        "meta_alfabetizacao_2024": "meta_alfabetizacao_ano_anterior",
    })

    enriquecido = alunos_presentes.merge(
        territorio_renomeado, on=["id_municipio", "ano_anterior"], how="left",
    )
    return enriquecido[COLUNAS_FINAIS_ENRIQUECIDO].reset_index(drop=True)


def materializar_tabela_enriquecida(client) -> None:
    """Consulta as fontes no BigQuery, aplica o enriquecimento, e grava o
    resultado em TABELA_DESTINO (substitui o conteúdo anterior)."""
    from google.cloud import bigquery  # import local: só necessário aqui, não nos testes

    alunos = client.query(f"""
        SELECT id_aluno, ano, id_municipio, rede, presenca, alfabetizado
        FROM `{TABELA_ALUNOS_FONTE}`
    """).to_dataframe()

    indicador_municipio = client.query(f"""
        SELECT id_municipio, ano, rede, sigla_uf,
               taxa_alfabetizacao, gap_meta_resultado, meta_alfabetizacao_2024
        FROM `{TABELA_INDICADOR_MUNICIPIO}`
    """).to_dataframe()

    territorio_por_ano = preparar_territorio_por_ano(indicador_municipio)
    enriquecido = enriquecer_alunos(alunos, territorio_por_ano)

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(enriquecido, TABELA_DESTINO, job_config=job_config)
    job.result()
```

- [ ] **Step 5: Rodar os testes e confirmar que passam**

Run: `venv/bin/python3 -m pytest tests/preprocessing/test_gold_materialization.py -v`
Expected: PASS (4 testes)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt conftest.py src/preprocessing/gold_materialization.py tests/preprocessing/test_gold_materialization.py
git commit -m "feat: materializa tabela Gold enriquecida (aluno + território defasado)"
```

---

## Task 2: Features e pré-processamento (`ColumnTransformer`)

**Files:**
- Create: `src/preprocessing/features.py`
- Test: `tests/preprocessing/test_features.py`

**Interfaces:**
- Consumes: colunas produzidas por `enriquecer_alunos` (Task 1) —
  `alfabetizado`, `rede`, `regiao`, `sigla_uf`,
  `taxa_alfabetizacao_ano_anterior`, `gap_meta_resultado_ano_anterior`,
  `meta_alfabetizacao_ano_anterior`.
- Produces:
  - `NUMERIC_FEATURES: list[str]`
  - `CATEGORICAL_FEATURES: list[str]`
  - `preparar_target(df: pd.DataFrame) -> pd.Series` — retorna a coluna
    `em_risco` (1 = não alfabetizado, 0 = alfabetizado).
  - `build_preprocessor() -> sklearn.compose.ColumnTransformer` — usado por
    Task 4 (`construir_candidatos`).

- [ ] **Step 1: Escrever os testes (falhando)**

Criar `tests/preprocessing/test_features.py`:

```python
import numpy as np
import pandas as pd

from src.preprocessing.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    preparar_target,
)


def test_preparar_target_inverte_alfabetizado():
    df = pd.DataFrame({"alfabetizado": [1, 0, 1, 1]})
    resultado = preparar_target(df)
    assert list(resultado) == [0, 1, 0, 0]
    assert resultado.name != "alfabetizado"


def _dados_treino():
    return pd.DataFrame({
        "taxa_alfabetizacao_ano_anterior": [70.0, np.nan, 80.0, 65.0],
        "gap_meta_resultado_ano_anterior": [-5.0, -2.0, np.nan, 3.0],
        "meta_alfabetizacao_ano_anterior": [80.0, 80.0, 80.0, 80.0],
        "rede": ["Municipal", "Estadual", "Municipal", "Privada"],
        "regiao": ["Sudeste", "Sudeste", "Nordeste", "Sul"],
        "sigla_uf": ["SP", "SP", "BA", "PR"],
    })


def test_build_preprocessor_imputa_nulos_e_sinaliza_ausencia():
    preprocessador = build_preprocessor()
    transformado = preprocessador.fit_transform(_dados_treino())

    assert not np.isnan(transformado).any()

    nomes = list(preprocessador.get_feature_names_out())
    assert any("missingindicator" in nome for nome in nomes)


def test_build_preprocessor_ignora_categoria_desconhecida_no_transform():
    preprocessador = build_preprocessor()
    preprocessador.fit(_dados_treino())

    dados_novos = pd.DataFrame({
        "taxa_alfabetizacao_ano_anterior": [50.0],
        "gap_meta_resultado_ano_anterior": [1.0],
        "meta_alfabetizacao_ano_anterior": [80.0],
        "rede": ["Federal"],  # categoria não vista no treino
        "regiao": ["Norte"],  # categoria não vista no treino
        "sigla_uf": ["AC"],   # categoria não vista no treino
    })

    # Não deve levantar exceção (handle_unknown='ignore').
    transformado = preprocessador.transform(dados_novos)
    assert transformado.shape[0] == 1


def test_numeric_and_categorical_features_batem_com_o_esperado():
    assert NUMERIC_FEATURES == [
        "taxa_alfabetizacao_ano_anterior",
        "gap_meta_resultado_ano_anterior",
        "meta_alfabetizacao_ano_anterior",
    ]
    assert CATEGORICAL_FEATURES == ["rede", "regiao", "sigla_uf"]
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `venv/bin/python3 -m pytest tests/preprocessing/test_features.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/preprocessing/features.py`**

```python
"""Definição de features e pré-processamento integrado (SimpleImputer +
OneHotEncoder dentro de um ColumnTransformer) — ver
docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "taxa_alfabetizacao_ano_anterior",
    "gap_meta_resultado_ano_anterior",
    "meta_alfabetizacao_ano_anterior",
]
CATEGORICAL_FEATURES = ["rede", "regiao", "sigla_uf"]


def preparar_target(df: pd.DataFrame) -> pd.Series:
    """Deriva 'em_risco' (1 = não alfabetizado, 0 = alfabetizado).

    Convenção usada em toda avaliação do projeto: a classe positiva (1)
    é a classe de interesse do negócio (risco), alinhando com o
    comportamento padrão do scikit-learn (predict_proba[:, 1],
    average_precision_score, precision_recall_curve) sem precisar de
    'pos_label' explícito em cada chamada.
    """
    return (1 - df["alfabetizado"]).rename("em_risco")


def build_preprocessor() -> ColumnTransformer:
    """Pré-processamento a ser integrado ao modelo via sklearn Pipeline
    (nunca aplicado manualmente antes do fit — ver Task 4)."""
    transformador_numerico = SimpleImputer(strategy="median", add_indicator=True)
    transformador_categorico = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("numericas", transformador_numerico, NUMERIC_FEATURES),
        ("categoricas", transformador_categorico, CATEGORICAL_FEATURES),
    ])
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `venv/bin/python3 -m pytest tests/preprocessing/test_features.py -v`
Expected: PASS (4 testes)

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/features.py tests/preprocessing/test_features.py
git commit -m "feat: define features e pré-processamento (imputação + one-hot)"
```

---

## Task 3: Split treino / validação / teste (temporal)

**Files:**
- Create: `src/preprocessing/splitting.py`
- Test: `tests/preprocessing/test_splitting.py`

**Interfaces:**
- Consumes: DataFrame com colunas `ano` e `em_risco` (produzida combinando
  `enriquecer_alunos` + `preparar_target`).
- Produces:
  - `dividir_treino_validacao_teste(df, coluna_ano="ano", coluna_target="em_risco", ano_desenvolvimento=2023, ano_teste_futuro=2024, random_state=42) -> dict[str, pd.DataFrame]`
    com chaves `"treino"`, `"validacao"`, `"teste_mesmo_ano"`, `"teste_futuro"`.

- [ ] **Step 1: Escrever os testes (falhando)**

Criar `tests/preprocessing/test_splitting.py`:

```python
import numpy as np
import pandas as pd

from src.preprocessing.splitting import dividir_treino_validacao_teste


def _dados_sinteticos(n_2023=2000, n_2024=500, taxa_risco=0.4, seed=0):
    rng = np.random.default_rng(seed)
    ano = [2023] * n_2023 + [2024] * n_2024
    em_risco = rng.binomial(1, taxa_risco, size=n_2023 + n_2024)
    return pd.DataFrame({"ano": ano, "em_risco": em_risco, "id": range(n_2023 + n_2024)})


def test_proporcoes_do_split_dentro_de_2023():
    df = _dados_sinteticos()
    partes = dividir_treino_validacao_teste(df)

    total_2023 = len(df[df["ano"] == 2023])
    assert abs(len(partes["treino"]) / total_2023 - 0.70) < 0.02
    assert abs(len(partes["validacao"]) / total_2023 - 0.15) < 0.02
    assert abs(len(partes["teste_mesmo_ano"]) / total_2023 - 0.15) < 0.02


def test_teste_futuro_e_2024_inteiro_e_nao_e_dividido():
    df = _dados_sinteticos()
    partes = dividir_treino_validacao_teste(df)

    assert (partes["teste_futuro"]["ano"] == 2024).all()
    assert len(partes["teste_futuro"]) == len(df[df["ano"] == 2024])


def test_estratificacao_preserva_balanceamento_da_classe():
    df = _dados_sinteticos(taxa_risco=0.4)
    partes = dividir_treino_validacao_teste(df)
    taxa_original = df[df["ano"] == 2023]["em_risco"].mean()

    for nome in ["treino", "validacao", "teste_mesmo_ano"]:
        taxa_parte = partes[nome]["em_risco"].mean()
        assert abs(taxa_parte - taxa_original) < 0.03, nome


def test_nenhuma_linha_de_2023_aparece_em_mais_de_uma_parte():
    df = _dados_sinteticos()
    partes = dividir_treino_validacao_teste(df)

    ids_treino = set(partes["treino"]["id"])
    ids_validacao = set(partes["validacao"]["id"])
    ids_teste = set(partes["teste_mesmo_ano"]["id"])

    assert ids_treino.isdisjoint(ids_validacao)
    assert ids_treino.isdisjoint(ids_teste)
    assert ids_validacao.isdisjoint(ids_teste)
    assert len(ids_treino | ids_validacao | ids_teste) == len(df[df["ano"] == 2023])
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `venv/bin/python3 -m pytest tests/preprocessing/test_splitting.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/preprocessing/splitting.py`**

```python
"""Split temporal (out-of-time validation) — ver
docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md, seção 'Split'."""

import pandas as pd
from sklearn.model_selection import train_test_split


def dividir_treino_validacao_teste(
    df: pd.DataFrame,
    coluna_ano: str = "ano",
    coluna_target: str = "em_risco",
    ano_desenvolvimento: int = 2023,
    ano_teste_futuro: int = 2024,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Dentro de `ano_desenvolvimento`: split fixo 70/15/15 (treino/validação/
    teste-mesmo-ano), estratificado por `coluna_target`. `ano_teste_futuro`
    inteiro vira o teste out-of-time, nunca dividido."""
    dev = df[df[coluna_ano] == ano_desenvolvimento]
    futuro = df[df[coluna_ano] == ano_teste_futuro]

    treino, resto = train_test_split(
        dev, test_size=0.30, stratify=dev[coluna_target], random_state=random_state,
    )
    validacao, teste_mesmo_ano = train_test_split(
        resto, test_size=0.50, stratify=resto[coluna_target], random_state=random_state,
    )

    return {
        "treino": treino.reset_index(drop=True),
        "validacao": validacao.reset_index(drop=True),
        "teste_mesmo_ano": teste_mesmo_ano.reset_index(drop=True),
        "teste_futuro": futuro.reset_index(drop=True),
    }
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `venv/bin/python3 -m pytest tests/preprocessing/test_splitting.py -v`
Expected: PASS (4 testes)

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/splitting.py tests/preprocessing/test_splitting.py
git commit -m "feat: split temporal treino/validação/teste (out-of-time em 2024)"
```

---

## Task 4: Candidatos de modelo (Pipelines integradas)

**Files:**
- Create: `src/modeling/candidates.py`
- Test: `tests/modeling/test_candidates.py`

**Interfaces:**
- Consumes: `build_preprocessor()` (Task 2).
- Produces:
  - `construir_candidatos(preprocessador: ColumnTransformer) -> dict[str, sklearn.pipeline.Pipeline]`
    com chaves `"regressao_logistica"`, `"random_forest"`,
    `"hist_gradient_boosting"`.

- [ ] **Step 1: Escrever os testes (falhando)**

Criar `tests/modeling/test_candidates.py`:

```python
import numpy as np
import pandas as pd

from src.preprocessing.features import build_preprocessor
from src.modeling.candidates import construir_candidatos


def _dados(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "taxa_alfabetizacao_ano_anterior": rng.normal(70, 10, n),
        "gap_meta_resultado_ano_anterior": rng.normal(0, 5, n),
        "meta_alfabetizacao_ano_anterior": rng.normal(80, 5, n),
        "rede": rng.choice(["Municipal", "Estadual"], n),
        "regiao": rng.choice(["Sudeste", "Nordeste"], n),
        "sigla_uf": rng.choice(["SP", "BA"], n),
    })


def test_construir_candidatos_retorna_as_tres_pipelines_esperadas():
    candidatos = construir_candidatos(build_preprocessor())
    assert set(candidatos) == {"regressao_logistica", "random_forest", "hist_gradient_boosting"}
    for pipeline in candidatos.values():
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict_proba")


def test_cada_pipeline_treina_e_preve_sem_erro():
    X = _dados()
    y = np.random.default_rng(1).integers(0, 2, len(X))
    candidatos = construir_candidatos(build_preprocessor())

    for nome, pipeline in candidatos.items():
        pipeline.fit(X, y)
        probabilidades = pipeline.predict_proba(X)
        assert probabilidades.shape == (len(X), 2), nome


def test_pipelines_nao_compartilham_estado_do_preprocessador():
    """Regressão: usar a MESMA instância de ColumnTransformer em 3 Pipelines
    sem clone() faz o fit de uma sobrescrever o estado das outras."""
    preprocessador_base = build_preprocessor()
    candidatos = construir_candidatos(preprocessador_base)

    X_a = _dados(n=30, seed=10)
    X_a["taxa_alfabetizacao_ano_anterior"] = 20.0  # mediana bem distinta de X_b
    y_a = np.random.default_rng(10).integers(0, 2, len(X_a))

    X_b = _dados(n=30, seed=20)
    X_b["taxa_alfabetizacao_ano_anterior"] = 90.0
    y_b = np.random.default_rng(20).integers(0, 2, len(X_b))

    candidatos["regressao_logistica"].fit(X_a, y_a)
    mediana_apos_fit_a = (
        candidatos["regressao_logistica"]
        .named_steps["preprocessamento"]
        .named_transformers_["numericas"]
        .statistics_[0]
    )

    candidatos["random_forest"].fit(X_b, y_b)
    mediana_lida_de_novo_em_a = (
        candidatos["regressao_logistica"]
        .named_steps["preprocessamento"]
        .named_transformers_["numericas"]
        .statistics_[0]
    )

    assert mediana_apos_fit_a == mediana_lida_de_novo_em_a
    assert mediana_apos_fit_a == 20.0
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `venv/bin/python3 -m pytest tests/modeling/test_candidates.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/modeling/candidates.py`**

```python
"""Candidatos de algoritmo, cada um como Pipeline (pré-processamento +
classificador) — ver docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md,
seção 'Modelo e seleção'."""

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def construir_candidatos(preprocessador: ColumnTransformer) -> dict[str, Pipeline]:
    """Cada Pipeline recebe seu próprio clone do preprocessador — evita que o
    fit de uma pipeline sobrescreva o estado ajustado (mediana, categorias)
    de outra, já que as três compartilhariam a mesma instância em memória
    se `clone()` não fosse usado aqui."""
    return {
        "regressao_logistica": Pipeline([
            ("preprocessamento", clone(preprocessador)),
            ("classificador", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessamento", clone(preprocessador)),
            ("classificador", RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocessamento", clone(preprocessador)),
            ("classificador", HistGradientBoostingClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
    }
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `venv/bin/python3 -m pytest tests/modeling/test_candidates.py -v`
Expected: PASS (3 testes)

- [ ] **Step 5: Commit**

```bash
git add src/modeling/candidates.py tests/modeling/test_candidates.py
git commit -m "feat: candidatos de modelo (regressão logística, random forest, hist gradient boosting)"
```

---

## Task 5: Seleção de modelo por validação (PR-AUC)

**Files:**
- Create: `src/modeling/selection.py`
- Test: `tests/modeling/test_selection.py`

**Interfaces:**
- Consumes: qualquer objeto com `.fit(X, y)` e `.predict_proba(X)` (as
  Pipelines de `construir_candidatos`, Task 4).
- Produces:
  - `selecionar_melhor_modelo(candidatos: dict[str, Any], X_treino, y_treino, X_validacao, y_validacao) -> tuple[str, Any, pd.DataFrame]`
    — retorna `(nome_vencedor, pipeline_vencedora, tabela_resultados)`,
    `tabela_resultados` com colunas `modelo`, `pr_auc_validacao`, ordenada
    desc.

- [ ] **Step 1: Escrever os testes (falhando)**

Criar `tests/modeling/test_selection.py`:

```python
import numpy as np
import pandas as pd

from src.modeling.selection import selecionar_melhor_modelo


class _PipelineFalsa:
    """Duck-type mínimo de uma Pipeline treinada: só precisa de fit/predict_proba.
    Usada para testar a LÓGICA de seleção isolada do comportamento real de
    um algoritmo de ML."""

    def __init__(self, probabilidades_fixas):
        self._probabilidades_fixas = np.asarray(probabilidades_fixas)
        self.foi_treinada = False

    def fit(self, X, y):
        self.foi_treinada = True
        return self

    def predict_proba(self, X):
        p = self._probabilidades_fixas
        return np.column_stack([1 - p, p])


def test_selecionar_melhor_modelo_escolhe_maior_pr_auc_na_validacao():
    y_validacao = pd.Series([0, 0, 1, 1, 1])

    candidatos = {
        "perfeito": _PipelineFalsa([0.01, 0.02, 0.9, 0.95, 0.99]),   # combina com y quase exatamente
        "aleatorio": _PipelineFalsa([0.5, 0.5, 0.5, 0.5, 0.5]),      # não discrimina nada
    }

    X_treino = pd.DataFrame({"x": range(10)})
    y_treino = pd.Series([0, 1] * 5)
    X_validacao = pd.DataFrame({"x": range(5)})

    nome_vencedor, pipeline_vencedora, tabela = selecionar_melhor_modelo(
        candidatos, X_treino, y_treino, X_validacao, y_validacao,
    )

    assert nome_vencedor == "perfeito"
    assert pipeline_vencedora is candidatos["perfeito"]
    assert candidatos["perfeito"].foi_treinada
    assert candidatos["aleatorio"].foi_treinada  # todos os candidatos são treinados

    assert list(tabela["modelo"]) == ["perfeito", "aleatorio"]
    assert tabela.iloc[0]["pr_auc_validacao"] > tabela.iloc[1]["pr_auc_validacao"]
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `venv/bin/python3 -m pytest tests/modeling/test_selection.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/modeling/selection.py`**

```python
"""Seleção de modelo via validação — critério: PR-AUC (average_precision_score)
da classe 'em_risco'. Ver docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md,
seção 'Modelo e seleção'."""

from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score


def selecionar_melhor_modelo(
    candidatos: dict[str, Any],
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_validacao: pd.DataFrame,
    y_validacao: pd.Series,
) -> tuple[str, Any, pd.DataFrame]:
    resultados = []
    for nome, pipeline in candidatos.items():
        pipeline.fit(X_treino, y_treino)
        probabilidades = pipeline.predict_proba(X_validacao)[:, 1]
        pr_auc = average_precision_score(y_validacao, probabilidades)
        resultados.append({"modelo": nome, "pr_auc_validacao": pr_auc})

    tabela_resultados = (
        pd.DataFrame(resultados)
        .sort_values("pr_auc_validacao", ascending=False)
        .reset_index(drop=True)
    )
    nome_vencedor = tabela_resultados.iloc[0]["modelo"]
    return nome_vencedor, candidatos[nome_vencedor], tabela_resultados
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `venv/bin/python3 -m pytest tests/modeling/test_selection.py -v`
Expected: PASS (1 teste)

- [ ] **Step 5: Commit**

```bash
git add src/modeling/selection.py tests/modeling/test_selection.py
git commit -m "feat: seleção de modelo vencedor por PR-AUC na validação"
```

---

## Task 6: Métricas de teste e tabela de limiares

**Files:**
- Create: `src/evaluation/metricas.py`
- Test: `tests/evaluation/test_metricas.py`

**Interfaces:**
- Consumes: pipeline treinada com `.predict_proba(X)` (Task 4/5).
- Produces:
  - `calcular_metricas_teste(pipeline, X, y) -> dict` com chaves
    `"roc_auc"`, `"pr_auc"`.
  - `tabela_limiares(pipeline, X, y, recall_alvo: float = 0.8) -> pd.DataFrame`
    com colunas `cenario`, `limiar`, `precisao`, `recall` e 3 linhas:
    `"padrao (limiar 0.5)"`, `"otimizado para F1"`,
    `"recall-prioritario (recall >= {alvo}%)"`.

- [ ] **Step 1: Escrever os testes (falhando)**

Criar `tests/evaluation/test_metricas.py`:

```python
import numpy as np

from src.evaluation.metricas import calcular_metricas_teste, tabela_limiares


class _PipelineFalsa:
    def __init__(self, probabilidades):
        self._probabilidades = np.asarray(probabilidades)

    def predict_proba(self, X):
        p = self._probabilidades
        return np.column_stack([1 - p, p])


def test_calcular_metricas_teste_com_separacao_perfeita():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    probabilidades = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95])
    pipeline = _PipelineFalsa(probabilidades)

    metricas = calcular_metricas_teste(pipeline, X=None, y=y)

    assert metricas["roc_auc"] == 1.0
    assert metricas["pr_auc"] == 1.0


def test_tabela_limiares_encontra_o_limiar_perfeito_para_f1():
    # y separa perfeitamente em 0.4: scores < 0.4 -> 0, scores >= 0.4 -> 1
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    probabilidades = np.array([0.1, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 0.9])
    pipeline = _PipelineFalsa(probabilidades)

    tabela = tabela_limiares(pipeline, X=None, y=y, recall_alvo=0.8)

    linha_f1 = tabela.set_index("cenario").loc["otimizado para F1"]
    assert linha_f1["precisao"] == 1.0
    assert linha_f1["recall"] == 1.0
    assert linha_f1["limiar"] == 0.4

    linha_recall_alvo = tabela.set_index("cenario").loc["recall-prioritario (recall >= 80%)"]
    assert linha_recall_alvo["recall"] >= 0.8

    linha_padrao = tabela.set_index("cenario").loc["padrao (limiar 0.5)"]
    assert linha_padrao["limiar"] == 0.5
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `venv/bin/python3 -m pytest tests/evaluation/test_metricas.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/evaluation/metricas.py`**

```python
"""Métricas de teste e tabela de limiares nomeados — ver
docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md, seção
'Avaliação e relato de métricas'. Convenção: y usa 'em_risco' (1 = classe
de interesse)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def calcular_metricas_teste(pipeline, X, y) -> dict:
    probabilidades = pipeline.predict_proba(X)[:, 1]
    return {
        "roc_auc": roc_auc_score(y, probabilidades),
        "pr_auc": average_precision_score(y, probabilidades),
    }


def _precisao_recall_em_limiar(y, probabilidades, limiar: float) -> tuple[float, float]:
    predito = (probabilidades >= limiar).astype(int)
    y = np.asarray(y)
    verdadeiro_positivo = int(((predito == 1) & (y == 1)).sum())
    falso_positivo = int(((predito == 1) & (y == 0)).sum())
    falso_negativo = int(((predito == 0) & (y == 1)).sum())
    precisao = verdadeiro_positivo / (verdadeiro_positivo + falso_positivo) if (verdadeiro_positivo + falso_positivo) else 0.0
    recall = verdadeiro_positivo / (verdadeiro_positivo + falso_negativo) if (verdadeiro_positivo + falso_negativo) else 0.0
    return precisao, recall


def tabela_limiares(pipeline, X, y, recall_alvo: float = 0.8) -> pd.DataFrame:
    probabilidades = pipeline.predict_proba(X)[:, 1]
    precisao, recall, limiares = precision_recall_curve(y, probabilidades)
    # precisao/recall têm 1 elemento a mais que limiares (o último ponto, sem
    # limiar correspondente, representa 'classificar tudo como negativo').
    precisao, recall = precisao[:-1], recall[:-1]

    f1 = 2 * precisao * recall / np.clip(precisao + recall, 1e-12, None)
    idx_f1 = int(np.argmax(f1))

    indices_recall_ok = np.where(recall >= recall_alvo)[0]
    # Entre os limiares que ainda satisfazem o recall mínimo, o de maior
    # índice é o de maior limiar (mais restritivo), que maximiza a precisão
    # sem violar o piso de recall.
    idx_recall_alvo = int(indices_recall_ok.max()) if len(indices_recall_ok) > 0 else int(np.argmax(recall))

    precisao_padrao, recall_padrao = _precisao_recall_em_limiar(y, probabilidades, 0.5)

    return pd.DataFrame([
        {"cenario": "padrao (limiar 0.5)", "limiar": 0.5, "precisao": precisao_padrao, "recall": recall_padrao},
        {"cenario": "otimizado para F1", "limiar": float(limiares[idx_f1]), "precisao": float(precisao[idx_f1]), "recall": float(recall[idx_f1])},
        {
            "cenario": f"recall-prioritario (recall >= {recall_alvo:.0%})",
            "limiar": float(limiares[idx_recall_alvo]),
            "precisao": float(precisao[idx_recall_alvo]),
            "recall": float(recall[idx_recall_alvo]),
        },
    ])
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `venv/bin/python3 -m pytest tests/evaluation/test_metricas.py -v`
Expected: PASS (2 testes)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metricas.py tests/evaluation/test_metricas.py
git commit -m "feat: métricas de teste e tabela de limiares nomeados (F1, recall-prioritário)"
```

---

## Task 7: Checks de variação temporal (covariate/concept shift)

**Files:**
- Create: `src/evaluation/variacao_temporal.py`
- Test: `tests/evaluation/test_variacao_temporal.py`

**Interfaces:**
- Consumes: dois DataFrames (dados de 2023 e 2024, já enriquecidos).
- Produces:
  - `comparar_distribuicoes_numericas(df_2023, df_2024, colunas: list[str]) -> pd.DataFrame`
    com colunas `coluna`, `estatistica_ks`, `p_valor`, `distribuicao_mudou`.
  - `comparar_distribuicoes_categoricas(df_2023, df_2024, colunas: list[str]) -> pd.DataFrame`
    com colunas `coluna`, `categoria`, `proporcao_2023`, `proporcao_2024`,
    `diferenca_absoluta`.

- [ ] **Step 1: Escrever os testes (falhando)**

Criar `tests/evaluation/test_variacao_temporal.py`:

```python
import numpy as np
import pandas as pd

from src.evaluation.variacao_temporal import (
    comparar_distribuicoes_categoricas,
    comparar_distribuicoes_numericas,
)


def test_comparar_distribuicoes_numericas_detecta_distribuicao_estavel():
    rng = np.random.default_rng(0)
    df_2023 = pd.DataFrame({"x": rng.normal(70, 5, 2000)})
    df_2024 = pd.DataFrame({"x": rng.normal(70, 5, 2000)})

    resultado = comparar_distribuicoes_numericas(df_2023, df_2024, ["x"])

    linha = resultado.set_index("coluna").loc["x"]
    assert linha["p_valor"] > 0.05
    assert linha["distribuicao_mudou"] is np.False_ or linha["distribuicao_mudou"] is False


def test_comparar_distribuicoes_numericas_detecta_mudanca_real():
    rng = np.random.default_rng(0)
    df_2023 = pd.DataFrame({"x": rng.normal(70, 5, 2000)})
    df_2024 = pd.DataFrame({"x": rng.normal(50, 5, 2000)})  # deslocamento grande

    resultado = comparar_distribuicoes_numericas(df_2023, df_2024, ["x"])

    linha = resultado.set_index("coluna").loc["x"]
    assert linha["p_valor"] < 0.05
    assert bool(linha["distribuicao_mudou"]) is True


def test_comparar_distribuicoes_categoricas_calcula_proporcoes():
    df_2023 = pd.DataFrame({"rede": ["Municipal"] * 80 + ["Estadual"] * 20})
    df_2024 = pd.DataFrame({"rede": ["Municipal"] * 50 + ["Estadual"] * 50})

    resultado = comparar_distribuicoes_categoricas(df_2023, df_2024, ["rede"])
    resultado = resultado.set_index("categoria")

    assert abs(resultado.loc["Municipal", "proporcao_2023"] - 0.8) < 1e-9
    assert abs(resultado.loc["Municipal", "proporcao_2024"] - 0.5) < 1e-9
    assert abs(resultado.loc["Municipal", "diferenca_absoluta"] - 0.3) < 1e-9
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `venv/bin/python3 -m pytest tests/evaluation/test_variacao_temporal.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/evaluation/variacao_temporal.py`**

```python
"""Diagnóstico de covariate shift vs. concept shift entre 2023 e 2024 — ver
docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md e
ensinamentos/02-modelagem/03-covariate-shift-vs-concept-shift.md."""

import pandas as pd
from scipy import stats


def comparar_distribuicoes_numericas(
    df_2023: pd.DataFrame, df_2024: pd.DataFrame, colunas: list[str],
) -> pd.DataFrame:
    linhas = []
    for coluna in colunas:
        valores_2023 = df_2023[coluna].dropna()
        valores_2024 = df_2024[coluna].dropna()
        estatistica, p_valor = stats.ks_2samp(valores_2023, valores_2024)
        linhas.append({
            "coluna": coluna,
            "estatistica_ks": estatistica,
            "p_valor": p_valor,
            "distribuicao_mudou": bool(p_valor < 0.05),
        })
    return pd.DataFrame(linhas)


def comparar_distribuicoes_categoricas(
    df_2023: pd.DataFrame, df_2024: pd.DataFrame, colunas: list[str],
) -> pd.DataFrame:
    linhas = []
    for coluna in colunas:
        proporcao_2023 = df_2023[coluna].value_counts(normalize=True)
        proporcao_2024 = df_2024[coluna].value_counts(normalize=True)
        categorias = sorted(set(proporcao_2023.index) | set(proporcao_2024.index))
        for categoria in categorias:
            p23 = float(proporcao_2023.get(categoria, 0.0))
            p24 = float(proporcao_2024.get(categoria, 0.0))
            linhas.append({
                "coluna": coluna,
                "categoria": categoria,
                "proporcao_2023": p23,
                "proporcao_2024": p24,
                "diferenca_absoluta": abs(p23 - p24),
            })
    return pd.DataFrame(linhas)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `venv/bin/python3 -m pytest tests/evaluation/test_variacao_temporal.py -v`
Expected: PASS (3 testes)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/variacao_temporal.py tests/evaluation/test_variacao_temporal.py
git commit -m "feat: checks de variação temporal entre 2023 e 2024 (KS test + proporções)"
```

---

## Task 8: Agregação de risco por município

**Files:**
- Create: `src/evaluation/risco_por_municipio.py`
- Test: `tests/evaluation/test_risco_por_municipio.py`

**Interfaces:**
- Consumes: `id_municipio` (Series) e `probabilidades` (array), tipicamente
  `pipeline.predict_proba(X)[:, 1]` do modelo vencedor (Task 5).
- Produces:
  - `agregar_risco_por_municipio(id_municipio: pd.Series, probabilidades) -> pd.DataFrame`
    com colunas `id_municipio`, `risco_medio`, `n_alunos`, ordenada desc por
    `risco_medio`.

- [ ] **Step 1: Escrever os testes (falhando)**

Criar `tests/evaluation/test_risco_por_municipio.py`:

```python
import numpy as np
import pandas as pd

from src.evaluation.risco_por_municipio import agregar_risco_por_municipio


def test_agrega_risco_medio_por_municipio_e_ordena_desc():
    id_municipio = pd.Series(["A", "A", "B", "B", "B"])
    probabilidades = np.array([0.2, 0.4, 0.9, 0.8, 0.7])

    resultado = agregar_risco_por_municipio(id_municipio, probabilidades)

    assert list(resultado["id_municipio"]) == ["B", "A"]
    assert abs(resultado.set_index("id_municipio").loc["A", "risco_medio"] - 0.3) < 1e-9
    assert resultado.set_index("id_municipio").loc["B", "n_alunos"] == 3
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `venv/bin/python3 -m pytest tests/evaluation/test_risco_por_municipio.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/evaluation/risco_por_municipio.py`**

```python
"""Agregação das predições por município — responde 'quais municípios
apresentam maior risco educacional' (ver spec, seção 'Perguntas de negócio
cobertas')."""

import numpy as np
import pandas as pd


def agregar_risco_por_municipio(id_municipio: pd.Series, probabilidades: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "id_municipio": pd.Series(id_municipio).values,
        "probabilidade_risco": np.asarray(probabilidades),
    })
    agregado = (
        df.groupby("id_municipio")["probabilidade_risco"]
        .agg(risco_medio="mean", n_alunos="count")
        .reset_index()
    )
    return agregado.sort_values("risco_medio", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `venv/bin/python3 -m pytest tests/evaluation/test_risco_por_municipio.py -v`
Expected: PASS (1 teste)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/risco_por_municipio.py tests/evaluation/test_risco_por_municipio.py
git commit -m "feat: agregação de risco médio por município"
```

---

## Task 9: Interpretabilidade (SHAP)

**Files:**
- Create: `src/modeling/interpretabilidade.py`
- Test: `tests/modeling/test_interpretabilidade.py`

**Interfaces:**
- Consumes: Pipeline treinada (Task 4), com `named_steps["preprocessamento"]`
  e `named_steps["classificador"]`.
- Produces:
  - `calcular_shap_values(pipeline, X_amostra: pd.DataFrame) -> shap.Explanation`
    — sempre 2D (`n_amostras, n_features`), `feature_names` preenchido.

- [ ] **Step 1: Escrever os testes (falhando)**

Criar `tests/modeling/test_interpretabilidade.py`:

```python
import numpy as np
import pandas as pd

from src.preprocessing.features import build_preprocessor
from src.modeling.candidates import construir_candidatos
from src.modeling.interpretabilidade import calcular_shap_values


def _dados(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "taxa_alfabetizacao_ano_anterior": rng.normal(70, 10, n),
        "gap_meta_resultado_ano_anterior": rng.normal(0, 5, n),
        "meta_alfabetizacao_ano_anterior": rng.normal(80, 5, n),
        "rede": rng.choice(["Municipal", "Estadual"], n),
        "regiao": rng.choice(["Sudeste", "Nordeste"], n),
        "sigla_uf": rng.choice(["SP", "BA"], n),
    })
    y = rng.integers(0, 2, n)
    return X, y


def test_calcular_shap_values_2d_para_cada_candidato():
    X, y = _dados()
    candidatos = construir_candidatos(build_preprocessor())

    for nome, pipeline in candidatos.items():
        pipeline.fit(X, y)
        valores_shap = calcular_shap_values(pipeline, X.iloc[:10])

        assert valores_shap.values.ndim == 2, nome
        assert valores_shap.values.shape[0] == 10, nome
        assert valores_shap.feature_names is not None, nome
        assert len(valores_shap.feature_names) == valores_shap.values.shape[1], nome
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `venv/bin/python3 -m pytest tests/modeling/test_interpretabilidade.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/modeling/interpretabilidade.py`**

```python
"""SHAP values do modelo vencedor — ver
docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md, seção
'Interpretabilidade'.

Nota: shap.TreeExplainer retorna formato diferente conforme o modelo —
RandomForestClassifier devolve um array 3D (amostras, features, classes),
enquanto HistGradientBoostingClassifier já devolve 2D. Este módulo
normaliza os dois casos para sempre retornar 2D (classe 'em_risco' = 1)."""

import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


def calcular_shap_values(pipeline, X_amostra: pd.DataFrame) -> shap.Explanation:
    preprocessador = pipeline.named_steps["preprocessamento"]
    classificador = pipeline.named_steps["classificador"]

    X_transformado = preprocessador.transform(X_amostra)
    nomes_features = list(preprocessador.get_feature_names_out())

    if isinstance(classificador, LogisticRegression):
        explicador = shap.LinearExplainer(classificador, X_transformado)
    else:
        explicador = shap.TreeExplainer(classificador)

    valores_shap = explicador(X_transformado)
    if valores_shap.values.ndim == 3:
        valores_shap = valores_shap[..., 1]  # classe 1 = em_risco

    valores_shap.feature_names = nomes_features
    return valores_shap
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `venv/bin/python3 -m pytest tests/modeling/test_interpretabilidade.py -v`
Expected: PASS (1 teste, cobrindo os 3 candidatos)

- [ ] **Step 5: Commit**

```bash
git add src/modeling/interpretabilidade.py tests/modeling/test_interpretabilidade.py
git commit -m "feat: cálculo de SHAP values normalizado entre os 3 candidatos"
```

---

## Task 10: Visualizações de suporte

**Files:**
- Create: `src/visualization/graficos.py`
- Test: `tests/evaluation/test_graficos.py`

**Interfaces:**
- Consumes: pipeline treinada + dados (Task 4/5); `nomes_features`/
  `importancias` (arrays paralelos, de `feature_importances_` ou
  `abs(coef_)` do modelo vencedor).
- Produces:
  - `plot_curva_precisao_recall(pipeline, X, y) -> matplotlib.figure.Figure`
  - `plot_importancia_features(nomes_features: list[str], importancias, top_n: int = 15) -> matplotlib.figure.Figure`

- [ ] **Step 1: Escrever os testes (falhando)**

Criar `tests/evaluation/test_graficos.py`:

```python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.graficos import plot_curva_precisao_recall, plot_importancia_features


class _PipelineFalsa:
    def __init__(self, probabilidades):
        self._probabilidades = np.asarray(probabilidades)

    def predict_proba(self, X):
        p = self._probabilidades
        return np.column_stack([1 - p, p])


def test_plot_curva_precisao_recall_retorna_figure():
    y = np.array([0, 0, 1, 1])
    pipeline = _PipelineFalsa([0.1, 0.4, 0.6, 0.9])

    fig = plot_curva_precisao_recall(pipeline, X=None, y=y)

    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_importancia_features_limita_ao_top_n():
    nomes = [f"feature_{i}" for i in range(20)]
    importancias = np.arange(20, dtype=float)

    fig = plot_importancia_features(nomes, importancias, top_n=5)

    eixo = fig.axes[0]
    assert len(eixo.patches) == 5
    plt.close(fig)


def test_plot_importancia_features_lida_com_menos_features_que_top_n():
    nomes = ["a", "b", "c"]
    importancias = np.array([1.0, 2.0, 3.0])

    fig = plot_importancia_features(nomes, importancias, top_n=15)

    eixo = fig.axes[0]
    assert len(eixo.patches) == 3
    plt.close(fig)
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `venv/bin/python3 -m pytest tests/evaluation/test_graficos.py -v`
Expected: FAIL com `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/visualization/graficos.py`**

```python
"""Gráficos de suporte para o notebook de modelagem e para o README."""

import matplotlib
matplotlib.use("Agg")  # evita exigir display gráfico em teste/CI

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


def plot_curva_precisao_recall(pipeline, X, y):
    probabilidades = pipeline.predict_proba(X)[:, 1]
    precisao, recall, _ = precision_recall_curve(y, probabilidades)

    fig, eixo = plt.subplots(figsize=(6, 5))
    eixo.plot(recall, precisao)
    eixo.set_xlabel("Recall")
    eixo.set_ylabel("Precisão")
    eixo.set_title("Curva Precisão-Recall (classe: em_risco)")
    fig.tight_layout()
    return fig


def plot_importancia_features(nomes_features, importancias, top_n: int = 15):
    importancias = np.asarray(importancias)
    top_n = min(top_n, len(nomes_features))
    indices_ordenados = np.argsort(importancias)[::-1][:top_n]

    nomes_top = [nomes_features[i] for i in indices_ordenados][::-1]
    valores_top = [importancias[i] for i in indices_ordenados][::-1]

    fig, eixo = plt.subplots(figsize=(8, max(3, top_n * 0.35)))
    eixo.barh(nomes_top, valores_top)
    eixo.set_xlabel("Importância")
    eixo.set_title(f"Top {top_n} features mais importantes")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `venv/bin/python3 -m pytest tests/evaluation/test_graficos.py -v`
Expected: PASS (3 testes)

- [ ] **Step 5: Commit**

```bash
git add src/visualization/graficos.py tests/evaluation/test_graficos.py
git commit -m "feat: gráficos de curva precisão-recall e importância de features"
```

---

## Task 11: Rodar a suíte completa e checar cobertura da spec

**Files:**
- Nenhum arquivo novo — task de verificação.

- [ ] **Step 1: Rodar toda a suíte de testes**

Run: `venv/bin/python3 -m pytest tests/ -v`
Expected: PASS — todos os testes das Tasks 1-10 (≈ 22 testes).

- [ ] **Step 2: Checar manualmente que cada peça da spec tem módulo correspondente**

Conferir, um a um, contra
`docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md`:

- Materialização da tabela Gold → `src/preprocessing/gold_materialization.py` ✓ (Task 1)
- Split treino/validação/teste → `src/preprocessing/splitting.py` ✓ (Task 3)
- Pré-processamento integrado (imputação + encoding) → `src/preprocessing/features.py` ✓ (Task 2)
- Candidatos + seleção por PR-AUC → `src/modeling/candidates.py` + `src/modeling/selection.py` ✓ (Tasks 4-5)
- Métricas de teste + tabela de limiares → `src/evaluation/metricas.py` ✓ (Task 6)
- Checks de variação temporal → `src/evaluation/variacao_temporal.py` ✓ (Task 7)
- Agregação de risco por município → `src/evaluation/risco_por_municipio.py` ✓ (Task 8)
- Interpretabilidade (SHAP) → `src/modeling/interpretabilidade.py` ✓ (Task 9)
- Visualizações → `src/visualization/graficos.py` ✓ (Task 10)

Se algo estiver faltando, é um sinal de que uma task foi pulada — não deve
acontecer se as Tasks 1-10 foram todas executadas.

- [ ] **Step 3: Commit (se houver qualquer ajuste feito durante a checagem)**

```bash
git status --short
# se houver mudanças pendentes de ajustes desta checagem:
git add -A
git commit -m "chore: ajustes finais de cobertura da suíte de testes da pipeline"
```

---

## Task 12: Notebook de orquestração (execução real contra BigQuery)

**Files:**
- Create: `notebooks/02_preprocessing_e_modelagem.ipynb`

**Interfaces:**
- Consumes: todos os módulos das Tasks 1-10.
- Produces: modelo treinado salvo (`joblib`), tabelas/gráficos usados no
  README (Task 13).

Este notebook não é testado por `pytest` (é execução real contra BigQuery,
fora do escopo de testes automatizados) — sua "verificação" é rodar
localmente do início ao fim sem erro e produzir os artefatos abaixo.

- [ ] **Step 1: Estrutura e materialização da tabela Gold**

Célula de setup (imports, cliente BigQuery), seguida de uma célula que roda,
**uma única vez** (não repetir a cada execução do notebook, é uma
materialização):

```python
from google.cloud import bigquery
from src.preprocessing.gold_materialization import materializar_tabela_enriquecida

client = bigquery.Client(project="fiapfase2")
materializar_tabela_enriquecida(client)
```

Célula markdown documentando (formato Motivo→Resultado→Decisão, mesmo do
notebook de EDA): por que essa tabela existe, o que ela contém, link para
`ensinamentos/02-modelagem/01-materializacao-de-camada-gold-versionada.md`.

- [ ] **Step 2: Carregar dado, preparar target, dividir splits**

```python
from src.preprocessing.features import preparar_target
from src.preprocessing.splitting import dividir_treino_validacao_teste

df = client.query("SELECT * FROM `fiapfase2.gold_alfabetizacao.aluno_alfabetizado_enriquecido`").to_dataframe()
df["em_risco"] = preparar_target(df)

partes = dividir_treino_validacao_teste(df)
X_treino, y_treino = partes["treino"], partes["treino"]["em_risco"]
X_validacao, y_validacao = partes["validacao"], partes["validacao"]["em_risco"]
X_teste_mesmo_ano, y_teste_mesmo_ano = partes["teste_mesmo_ano"], partes["teste_mesmo_ano"]["em_risco"]
X_teste_futuro, y_teste_futuro = partes["teste_futuro"], partes["teste_futuro"]["em_risco"]
```

Célula markdown: reportar `df.shape`, `df["em_risco"].mean()` geral e por
ano — comparar com os números já conhecidos da EDA (~41%/59%) como checagem
de sanidade.

- [ ] **Step 3: Treinar candidatos e selecionar vencedor**

```python
from src.preprocessing.features import build_preprocessor
from src.modeling.candidates import construir_candidatos
from src.modeling.selection import selecionar_melhor_modelo

candidatos = construir_candidatos(build_preprocessor())
nome_vencedor, modelo_vencedor, tabela_selecao = selecionar_melhor_modelo(
    candidatos, X_treino, y_treino, X_validacao, y_validacao,
)
display(tabela_selecao)
print("Modelo vencedor:", nome_vencedor)
```

Célula markdown: registrar qual modelo venceu e o PR-AUC de cada candidato
na validação — essa é a única vez que os números de validação entram no
relatório final (a partir daqui, só teste).

- [ ] **Step 4: Avaliação final (teste-2023 e 2024) — uma única vez**

```python
from src.evaluation.metricas import calcular_metricas_teste, tabela_limiares

metricas_teste_mesmo_ano = calcular_metricas_teste(modelo_vencedor, X_teste_mesmo_ano, y_teste_mesmo_ano)
metricas_teste_futuro = calcular_metricas_teste(modelo_vencedor, X_teste_futuro, y_teste_futuro)

print("Teste-2023 (mesmo ano):", metricas_teste_mesmo_ano)
print("2024 (out-of-time):", metricas_teste_futuro)

limiares_teste_futuro = tabela_limiares(modelo_vencedor, X_teste_futuro, y_teste_futuro)
display(limiares_teste_futuro)
```

Célula markdown: interpretar o gap entre teste-2023 e 2024 (ver
`ensinamentos/02-modelagem/02-split-temporal-out-of-time-validation.md`).

- [ ] **Step 5: Diagnóstico de variação temporal**

```python
from src.evaluation.variacao_temporal import comparar_distribuicoes_categoricas, comparar_distribuicoes_numericas
from src.preprocessing.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

df_2023 = df[df["ano"] == 2023]
df_2024 = df[df["ano"] == 2024]

display(comparar_distribuicoes_numericas(df_2023, df_2024, NUMERIC_FEATURES))
display(comparar_distribuicoes_categoricas(df_2023, df_2024, CATEGORICAL_FEATURES))
```

Célula markdown: usar esse resultado para classificar o gap da Etapa 4 como
covariate shift, concept shift, ou "sem variação relevante" (ver
`ensinamentos/02-modelagem/03-covariate-shift-vs-concept-shift.md`).

- [ ] **Step 6: Interpretabilidade — feature importance/coeficientes + SHAP**

```python
from src.modeling.interpretabilidade import calcular_shap_values
from src.visualization.graficos import plot_importancia_features, plot_curva_precisao_recall

preprocessador_ajustado = modelo_vencedor.named_steps["preprocessamento"]
classificador_vencedor = modelo_vencedor.named_steps["classificador"]
nomes_features = list(preprocessador_ajustado.get_feature_names_out())

if hasattr(classificador_vencedor, "feature_importances_"):
    importancias = classificador_vencedor.feature_importances_
else:
    importancias = abs(classificador_vencedor.coef_[0])

plot_importancia_features(nomes_features, importancias)
plot_curva_precisao_recall(modelo_vencedor, X_teste_futuro, y_teste_futuro)

# SHAP: amostra (custo computacional), não o dataset completo — ver
# ensinamentos/02-modelagem (nota sobre amostragem para SHAP na spec).
amostra_shap = X_teste_futuro.sample(n=min(3000, len(X_teste_futuro)), random_state=42)
valores_shap = calcular_shap_values(modelo_vencedor, amostra_shap)

import shap
shap.summary_plot(valores_shap, amostra_shap)
```

- [ ] **Step 7: Agregação de risco por município e salvar o modelo**

```python
import joblib
from src.evaluation.risco_por_municipio import agregar_risco_por_municipio

probabilidades_futuro = modelo_vencedor.predict_proba(X_teste_futuro)[:, 1]
risco_por_municipio = agregar_risco_por_municipio(X_teste_futuro["id_municipio"], probabilidades_futuro)
display(risco_por_municipio.head(20))

risco_por_municipio.to_csv("reports/risco_por_municipio_2024.csv", index=False)
joblib.dump(modelo_vencedor, "reports/modelo_vencedor.joblib")
```

Célula markdown final: síntese (modelo escolhido, métricas principais,
top-5 municípios de maior risco, top-5 features mais influentes) — isso
alimenta diretamente as seções do README na Task 13.

- [ ] **Step 8: Rodar o notebook do início ao fim e verificar**

Run: `venv/bin/jupyter nbconvert --to notebook --execute notebooks/02_preprocessing_e_modelagem.ipynb --output 02_preprocessing_e_modelagem.ipynb`
Expected: execução sem erro; `reports/risco_por_municipio_2024.csv` e
`reports/modelo_vencedor.joblib` criados.

- [ ] **Step 9: Commit**

```bash
git add notebooks/02_preprocessing_e_modelagem.ipynb
git commit -m "feat: notebook de preprocessing e modelagem ponta a ponta"
```

(`reports/*.csv` e `*.joblib` não são versionados se `reports/` estiver
coberto por uma regra de dados grandes no `.gitignore` — conferir antes do
commit; se não estiver, considerar adicionar `reports/*.joblib` e
`reports/*.csv` ao `.gitignore`, mantendo só `reports/.gitkeep`.)

---

## Task 13: Atualizar README com metodologia e resultados

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: resultados produzidos pela Task 12 (tabela de seleção,
  métricas de teste, tabela de limiares, top features, top municípios).

- [ ] **Step 1: Preencher "Etapas de modelagem" e "Escolha do algoritmo"**

Substituir os blocos `> _TODO_` correspondentes em `README.md` por um
resumo em português cobrindo: materialização da tabela Gold enriquecida,
split temporal (2023 treino/val/teste + 2024 out-of-time), pré-processamento
(imputação com indicador de ausência + one-hot), os 3 candidatos e o
critério de seleção (PR-AUC na validação) — linkando para
`docs/superpowers/specs/2026-09-11-pipeline-modelagem-design.md` como
referência completa.

- [ ] **Step 2: Preencher "Métricas de avaliação", "Interpretação dos resultados" e "Insights encontrados"**

Usar os números reais gerados pela Task 12 (ROC-AUC/PR-AUC teste-2023 vs.
2024, tabela de limiares, top features do SHAP, top municípios de risco,
resultado do diagnóstico covariate/concept shift). Não preencher com
números fictícios — se a Task 12 ainda não rodou contra o BigQuery real
quando esta task for executada, deixar essas 3 seções para depois da Task
12 rodar de verdade (não travar a Task 13 com números inventados).

- [ ] **Step 3: Preencher "Limitações do projeto" e "Possíveis evoluções futuras"**

Cobrir explicitamente: só 2 anos de dado (limita validação temporal a uma
única fronteira, não a uma série robusta — ver
`ensinamentos/02-modelagem/02-split-temporal-out-of-time-validation.md`),
sem fontes externas nesta rodada, limiar final de decisão não travado
(depende de capacidade operacional do gestor público), pergunta de negócio
"quais regiões têm padrões semelhantes" tratada como próximo passo
(clusterização, fora do escopo desta spec).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: preenche README com metodologia e resultados da pipeline de modelagem"
```

---

## Self-Review

**Cobertura da spec:** todas as seções de "Pré-processamento", "Modelo e
seleção", "Avaliação e relato de métricas", "Interpretabilidade" e as 3
perguntas de negócio marcadas com ✅ têm task correspondente (Tasks 1-10,
orquestradas na 12). As 2 itens marcados ⚠️/fora de escopo na spec
(clusterização regional, fontes externas, limiar único) permanecem
deliberadamente fora deste plano — documentado na Task 13, Step 3.

**Placeholders:** nenhum "TBD"/"implementar depois" nas Tasks 1-12; a única
ressalva explícita é a Task 13 Step 2, que depende de números reais gerados
pela Task 12 (não é um placeholder de código, é uma dependência de dado
real que não existe até a pipeline rodar).

**Consistência de tipos:** `em_risco` é o nome usado consistentemente do
Task 2 em diante; `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` (Task 2) são
reusadas sem redefinição nas Tasks 4, 9, 12; `TABELA_DESTINO` (Task 1) é a
única referência ao nome da tabela materializada, reusada na Task 12.
