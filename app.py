from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

from services.previsao import gerar_previsao, format_previsao_legacy

app = Flask(__name__)
CORS(app)

# Endpoints NDVI existentes
@app.route('/dados')
def get_dados():
    df = pd.read_csv('borneu_vegetacao_anual.csv')
    dados = df.to_dict(orient='records')
    return jsonify(dados)

@app.route('/previsao', methods=['GET'])
def previsao():
    df = pd.read_csv('borneu_vegetacao_anual.csv')
    resultado = gerar_previsao(df)

    # Query params
    fmt = request.args.get('format', 'legacy')
    combined = request.args.get('combined', 'false').lower() == 'true'

    if fmt == 'full':
        if combined:
            # concatena dados reais + previsões
            dados_reais = df.to_dict(orient='records')
            previsoes = format_previsao_legacy(resultado)
            # para distinguir, adiciona NDVI_previsto apenas nas previsões
            for d in dados_reais:
                d['NDVI_previsto'] = None
            dados_combinados = dados_reais + previsoes
            return jsonify(dados_combinados)
        return jsonify(resultado)

    # legacy: apenas previsões simples
    legacy = format_previsao_legacy(resultado)
    return jsonify(legacy)

if __name__ == '__main__':
    app.run(debug=True)
