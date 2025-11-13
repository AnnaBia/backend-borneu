import os
import xarray as xr
import rioxarray
import numpy as np

def carregar_dados_climaticos(pasta="borneu_clima"):
    """Lê todos os GeoTIFFs de temperatura e precipitação e calcula médias anuais."""
    dados = []

    # Detecta os anos disponíveis automaticamente
    anos = sorted(set(
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(pasta)
        if f.endswith(".tif")
    ))

    for ano in anos:
        temp_path = os.path.join(pasta, f"temp_{ano}.tif")
        precip_path = os.path.join(pasta, f"precip_{ano}.tif")

        temp_mean = np.nan
        precip_mean = np.nan

        if os.path.exists(temp_path):
            temp_ds = rioxarray.open_rasterio(temp_path)
            temp_mean = float(temp_ds.mean().values)

        if os.path.exists(precip_path):
            precip_ds = rioxarray.open_rasterio(precip_path)
            precip_mean = float(precip_ds.mean().values)

        dados.append({
            "year": ano,
            "temp": round(temp_mean, 2) if not np.isnan(temp_mean) else None,
            "precip": round(precip_mean, 2) if not np.isnan(precip_mean) else None
        })

    return dados
