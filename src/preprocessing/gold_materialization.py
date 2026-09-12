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
# rede). Escolhida como "Municipal" (não a mesma da EDA/H2) porque é a única
# rede com gap_meta_resultado e meta_alfabetizacao_2024 populadas no Gold real.
REDE_REFERENCIA_TERRITORIO = "Municipal"

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
    ANTERIOR ao ano do aluno quando esse ano existir na Gold — evita vazar o
    resultado do próprio período. Quando o ano anterior não existir na Gold
    (ex.: aluno de 2023 precisaria de território de 2022, que a Gold real
    não tem — só cobre 2023/2024), cai para o MESMO ano do aluno como
    fallback. Essa é uma concessão deliberada a uma limitação real de dado
    (só 2 anos existem no total): o vazamento potencial do fallback (usar
    o agregado municipal do mesmo ano/período) é diluído entre
    centenas/milhares de alunos do município naquele ano — categoricamente
    diferente do vazamento de `proficiencia` (definição determinística do
    target individual, ~100% de concordância). Só afeta o cohort de
    desenvolvimento (2023 nesta versão dos dados); o cohort de 2024 sempre
    usa o ano anterior de verdade (2023), sem fallback."""
    alunos_presentes = alunos[alunos["presenca"] == "1"].copy()
    alunos_presentes["rede"] = alunos_presentes["rede"].map(MAPEAMENTO_REDE)
    alunos_presentes["alfabetizado"] = alunos_presentes["alfabetizado"].astype(int)
    alunos_presentes["ano"] = alunos_presentes["ano"].astype("int64")

    ano_minimo_disponivel = territorio_por_ano["ano"].min()
    ano_anterior = alunos_presentes["ano"] - 1
    alunos_presentes["ano_referencia"] = ano_anterior.where(
        ano_anterior >= ano_minimo_disponivel, alunos_presentes["ano"]
    )

    territorio_renomeado = territorio_por_ano.rename(columns={
        "ano": "ano_referencia",
        "taxa_alfabetizacao": "taxa_alfabetizacao_ano_anterior",
        "gap_meta_resultado": "gap_meta_resultado_ano_anterior",
        "meta_alfabetizacao_2024": "meta_alfabetizacao_ano_anterior",
    })

    enriquecido = alunos_presentes.merge(
        territorio_renomeado, on=["id_municipio", "ano_referencia"], how="left",
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
