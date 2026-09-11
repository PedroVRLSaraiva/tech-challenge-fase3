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
