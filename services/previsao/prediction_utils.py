import numpy as np
from typing import List, Dict, Any
from .model_utils import treinar_modelo, calcular_metricas

def prever_anos(modelo, anos: List[int], resid_std: float, col_y: str) -> List[Dict[str, Any]]:
    """Prevê anos futuros com intervalo de confiança aproximado."""
    Xf = np.array(anos, dtype=float).reshape(-1, 1)
    preds = modelo.predict(Xf)
    ci_half = 1.96 * resid_std

    return [
        {
            'year': int(a),
            f'{col_y}_previsto': float(p),
            'lower_95': float(p - ci_half),
            'upper_95': float(p + ci_half),
        }
        for a, p in zip(anos, preds)
    ]


def gerar_previsao(df, col_y: str, start_year: int = 2023, end_year: int = 2030):
    """Gera previsão genérica para NDVI, temperatura, precipitação ou área."""
    modelo, X, y = treinar_modelo(df, col_y)
    mets = calcular_metricas(modelo, X, y)
    anos_futuros = list(range(start_year, end_year + 1))
    preds = prever_anos(modelo, anos_futuros, mets['residual_std'], col_y)

    return {
        'variavel': col_y,
        'modelo': 'linear_regression',
        'coef': float(modelo.coef_[0]),
        'intercept': float(modelo.intercept_),
        **mets,
        'train_years': X.flatten().astype(int).tolist(),
        'train_values': y.tolist(),
        'predictions': preds
    }
