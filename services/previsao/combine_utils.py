from typing import Dict, Any, Optional, List
import pandas as pd

def combinar_observados_previstos(resultado: Dict[str, Any], df: Optional[pd.DataFrame], col_y: str):
    """Combina observados + previstos para exibição no front."""
    preds = resultado.get('predictions', [])
    pred_map = {int(p['year']): p[f'{col_y}_previsto'] for p in preds}
    obs_map = {int(r['year']): r[col_y] for _, r in df[['year', col_y]].dropna().iterrows()}

    anos = sorted(set(obs_map.keys()) | set(pred_map.keys()))
    pontos = []
    for a in anos:
        if a in obs_map:
            pontos.append({'year': a, 'value': obs_map[a], 'type': 'observed'})
        elif a in pred_map:
            pontos.append({'year': a, 'value': pred_map[a], 'type': 'predicted'})

    return {'anos': anos, 'pontos': pontos}
