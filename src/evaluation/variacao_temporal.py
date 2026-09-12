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
