from sklearn.linear_model import LinearRegression
import numpy as np

def treinar_modelo(df, col_y):
    """Treina um modelo linear simples (ano → variável)"""
    df = df.dropna(subset=['year', col_y])
    X = df['year'].values.reshape(-1, 1)
    y = df[col_y].values
    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo, X, y


def calcular_metricas(modelo, X, y):
    """Calcula erro padrão residual (para IC 95%)."""
    from sklearn.metrics import mean_squared_error
    y_pred = modelo.predict(X)
    mse = mean_squared_error(y, y_pred)
    residual_std = np.sqrt(mse)
    return {
        'mse': float(mse),
        'residual_std': float(residual_std)
    }
