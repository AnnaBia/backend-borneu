from typing import Dict, List, Any, Optional, Tuple
import logging

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

logger = logging.getLogger(__name__)


def treinar_modelo(df: pd.DataFrame) -> Tuple[LinearRegression, np.ndarray, np.ndarray]:
    """Treina um modelo de regressão linear simples (year -> NDVI).

    Recebe um DataFrame com colunas 'year' e 'NDVI' e retorna o modelo treinado
    junto com X (anos) e y (valores NDVI) usados no treino.
    """
    if 'year' not in df.columns or 'NDVI' not in df.columns:
        raise ValueError("O DataFrame precisa conter as colunas 'year' e 'NDVI'")

    sub = df[['year', 'NDVI']].dropna()
    if len(sub) < 2:
        raise ValueError('Dados insuficientes para treinar o modelo (necessários >= 2 linhas)')

    X = sub['year'].astype(float).values.reshape(-1, 1)
    y = sub['NDVI'].astype(float).values

    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo, X, y


def calcular_metricas(modelo: LinearRegression, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de ajuste do modelo (R2, RMSE, std dos resíduos)."""
    y_pred = modelo.predict(X)
    r2 = float(r2_score(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))
    rmse = float(math.sqrt(mse))
    residuals = y - y_pred
    resid_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    return {'r2': r2, 'rmse': rmse, 'residual_std': resid_std}


def prever_anos(modelo: LinearRegression, anos: List[int], resid_std: float) -> List[Dict[str, Any]]:
    """Gera previsões para uma lista de anos e retorna uma lista de dicionários com IC aproximado."""
    Xf = np.array(anos, dtype=float).reshape(-1, 1)
    preds = modelo.predict(Xf)
    ci_half = 1.96 * resid_std
    result = []
    for ano, p in zip(anos, preds):
        result.append({
            'year': int(ano),
            'NDVI_previsto': float(p),
            'lower_95': float(p - ci_half),
            'upper_95': float(p + ci_half),
        })
    return result


def gerar_previsao(df: pd.DataFrame, start_year: int = 2023, end_year: int = 2030) -> Dict[str, Any]:
    """Interface principal: treina, calcula métricas e retorna previsões formatadas.

    Retorna um dicionário com chaves: model, coef, intercept, r2, rmse, residual_std,
    train_years, train_values, predictions.
    """
    modelo, X, y = treinar_modelo(df)
    mets = calcular_metricas(modelo, X, y)
    anos_futuros = list(range(start_year, end_year + 1))
    preds = prever_anos(modelo, anos_futuros, mets['residual_std'])

    resultado = {
        'model': 'linear_regression',
        'coef': float(modelo.coef_[0]),
        'intercept': float(modelo.intercept_),
        'r2': mets['r2'],
        'rmse': mets['rmse'],
        'residual_std': mets['residual_std'],
        'train_years': [int(v) for v in X.reshape(-1)],
        'train_values': [float(v) for v in y.tolist()],
        'predictions': preds,
    }
    logger.debug('Previsão gerada: %s', resultado)
    return resultado


def format_previsao_legacy(resultado: Any) -> List[Dict[str, Any]]:
    """Converte o resultado enriquecido para o formato legado (lista simples).

    Aceita tanto o dicionário novo (com 'predictions') quanto uma lista já no formato
    legado e devolve sempre a lista de previsões simples: [{'year':..., 'NDVI_previsto':...}, ...]
    """
    if resultado is None:
        return []
    if isinstance(resultado, list):
        return resultado
    if isinstance(resultado, dict) and 'predictions' in resultado:
        preds = resultado['predictions']
        out = []
        for p in preds:
            year = int(p.get('year'))
            ndvi = None
            if p.get('NDVI_previsto') is not None:
                ndvi = p.get('NDVI_previsto')
            elif p.get('NDVI') is not None:
                ndvi = p.get('NDVI')
            elif p.get('value') is not None:
                ndvi = p.get('value')
            out.append({'year': year, 'NDVI_previsto': float(ndvi) if ndvi is not None else None})
        return out
    return []


def build_combined_series(resultado: Any, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Cria série combinada (observados + previstos) pronta para plotagem.

    Retorna dict com years[], observed[], predicted[], lower_95[], upper_95[], points[]
    """
    # extrair previsões
    preds = []
    if isinstance(resultado, dict) and 'predictions' in resultado:
        preds = resultado['predictions']
    elif isinstance(resultado, list):
        preds = resultado

    pred_map = {int(p['year']): float(p.get('NDVI_previsto') or p.get('NDVI') or p.get('value')) for p in preds}
    lower_map = {int(p['year']): float(p.get('lower_95')) for p in preds if p.get('lower_95') is not None}
    upper_map = {int(p['year']): float(p.get('upper_95')) for p in preds if p.get('upper_95') is not None}

    obs_map = {}
    if df is not None:
        for _, row in df[['year', 'NDVI']].dropna().iterrows():
            yr = int(float(row['year']))
            obs_map[yr] = float(row['NDVI'])

    years = sorted(set(list(obs_map.keys()) + list(pred_map.keys())))
    observed = [obs_map.get(y) for y in years]
    predicted = [pred_map.get(y) if obs_map.get(y) is None else None for y in years]
    lower = [lower_map.get(y) if obs_map.get(y) is None else None for y in years]
    upper = [upper_map.get(y) if obs_map.get(y) is None else None for y in years]

    points = []
    for y, o, p, l, u in zip(years, observed, predicted, lower, upper):
        if o is not None:
            points.append({'year': int(y), 'value': o, 'type': 'observed'})
        if p is not None:
            pt = {'year': int(y), 'value': p, 'type': 'predicted'}
            if l is not None and u is not None:
                pt['lower'] = l
                pt['upper'] = u
            points.append(pt)

    combined = {
        'model': resultado.get('model') if isinstance(resultado, dict) else 'unknown',
        'coef': resultado.get('coef') if isinstance(resultado, dict) else None,
        'intercept': resultado.get('intercept') if isinstance(resultado, dict) else None,
        'r2': resultado.get('r2') if isinstance(resultado, dict) else None,
        'rmse': resultado.get('rmse') if isinstance(resultado, dict) else None,
        'years': years,
        'observed': observed,
        'predicted': predicted,
        'lower_95': lower,
        'upper_95': upper,
        'points': points,
    }
    return combined


def build_combined_records(resultado: Any, df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """Retorna lista de records semelhante ao `df.to_dict(orient='records')` com previsões anexadas.

    Observados são preservados; previsões viram novas linhas com 'predicted': True.
    """
    records: List[Dict[str, Any]] = []
    if df is not None:
        records = df.to_dict(orient='records')

    if isinstance(resultado, dict) and 'predictions' in resultado:
        preds = resultado['predictions']
    elif isinstance(resultado, list):
        preds = resultado
    else:
        preds = []

    next_idx = 0
    if records:
        try:
            idxs = [int(r.get('system:index')) for r in records if r.get('system:index') is not None]
            if idxs:
                next_idx = max(idxs) + 1
            else:
                next_idx = len(records)
        except Exception:
            next_idx = len(records)

    base_cols = list(records[0].keys()) if records else ['system:index', 'NDVI', 'year', '.geo']

    for p in preds:
        y = int(p.get('year'))
        nd = p.get('NDVI_previsto') if p.get('NDVI_previsto') is not None else p.get('NDVI') or p.get('value')
        rec: Dict[str, Any] = {}
        for c in base_cols:
            if c == 'system:index':
                rec[c] = next_idx
            elif c == 'NDVI':
                rec[c] = float(nd) if nd is not None else None
            elif c == 'year':
                rec[c] = int(y)
            elif c == '.geo':
                rec[c] = ''
            else:
                rec[c] = None
        rec['predicted'] = True
        records.append(rec)
        next_idx += 1
    return records


__all__ = [
    'gerar_previsao',
    'format_previsao_legacy',
    'build_combined_series',
    'build_combined_records',
]
