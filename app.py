from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd

from services.climate_service import carregar_dados_climaticos
from services.previsao.combine_utils import combinar_observados_previstos
from services.previsao.prediction_utils import gerar_previsao

app = Flask(__name__)
CORS(app)

# --- CLIMA (a partir dos arquivos .tif) ---
@app.route('/clima/extract')
def clima_extract():
    """Extrai e retorna dados climáticos (temperatura e precipitação)."""
    dados_clima = carregar_dados_climaticos()
    return jsonify(dados_clima)


# --- DESMATAMENTO (dados históricos por país) ---
@app.route('/desmatamento')
def get_desmatamento():
    """Retorna dados históricos de área desmatada."""
    df = pd.read_excel('borneu_area_desmatada.xlsx')

    # Normaliza colunas
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    df = df.rename(columns={
        'ano': 'year',
        'área_desmatada_km2': 'area_desmatada',
        'país': 'pais',
        'região': 'regiao'
    })

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['area_desmatada'] = pd.to_numeric(df['area_desmatada'], errors='coerce')
    df = df.dropna(subset=['year', 'area_desmatada'])
    df['year'] = df['year'].astype(int)

    # Agrupa por ano e país
    grouped = df.groupby(['year', 'pais'])['area_desmatada'].sum().unstack(fill_value=0)

    # Prepara formato para gráfico
    dados = {
        "years": grouped.index.tolist(),
        "series": [
            {"label": pais, "data": grouped[pais].tolist()}
            for pais in grouped.columns
        ]
    }

    return jsonify(dados)

@app.route('/ndvi')
def get_ndvi():
    """Retorna NDVI anual observado (histórico)."""
    df = pd.read_csv('borneu_vegetacao_anual.csv')
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Garante que as colunas estejam no formato esperado
    if 'year' not in df.columns or 'ndvi' not in df.columns:
        return jsonify({'erro': 'Arquivo NDVI inválido'}), 400

    dados = df[['year', 'ndvi']].dropna().to_dict(orient='records')
    return jsonify(dados)

@app.route('/previsao/clima')
def previsao_clima():
    """Gera previsões conjuntas de temperatura e precipitação até 2030."""
    dados_clima = carregar_dados_climaticos("borneu_clima")
    df = pd.DataFrame(dados_clima)

    # Previsões separadas para cada variável
    resultado_temp = gerar_previsao(df, 'temp', end_year=2030)
    resultado_precip = gerar_previsao(df, 'precip', end_year=2030)

    combinado_temp = combinar_observados_previstos(resultado_temp, df, 'temp')
    combinado_precip = combinar_observados_previstos(resultado_precip, df, 'precip')

    # Junta em um único formato
    anos = sorted(set(combinado_temp['anos']) | set(combinado_precip['anos']))
    resposta = []
    for ano in anos:
        temp = next((p['value'] for p in combinado_temp['pontos'] if p['year'] == ano), None)
        precip = next((p['value'] for p in combinado_precip['pontos'] if p['year'] == ano), None)
        resposta.append({
            'year': ano,
            'temp': temp,
            'precip': precip
        })

    return jsonify(resposta)


@app.route('/previsao/area_desmatada')
def previsao_area_desmatada():
    """Gera previsões de área desmatada por país até 2030."""
    arquivo = 'borneu_area_desmatada.xlsx'
    df = pd.read_excel(arquivo)

    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    df = df.rename(columns={'ano': 'year', 'área_desmatada_km2': 'area_desmatada', 'país': 'pais'})

    series = []
    anos_total = set()

    for pais, df_pais in df.groupby('pais'):
        resultado = gerar_previsao(df_pais, 'area_desmatada', end_year=2030)
        combinado = combinar_observados_previstos(resultado, df_pais, 'area_desmatada')

        anos = combinado['anos']
        anos_total.update(anos)
        valores = [p['value'] for p in combinado['pontos']]

        series.append({
            'label': pais,
            'data': valores
        })

    return jsonify({
        'years': sorted(list(anos_total)),
        'series': series
    })

@app.route('/previsao/ndvi')
def previsao_ndvi():
    """Gera previsão de NDVI até 2030."""
    df = pd.read_csv('borneu_vegetacao_anual.csv')
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    if 'year' not in df.columns or 'ndvi' not in df.columns:
        return jsonify({'erro': 'Arquivo NDVI inválido'}), 400

    resultado = gerar_previsao(df, 'ndvi', end_year=2030)
    combinado = combinar_observados_previstos(resultado, df, 'ndvi')

    return jsonify(combinado)

if __name__ == '__main__':
    app.run(debug=True)
