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
