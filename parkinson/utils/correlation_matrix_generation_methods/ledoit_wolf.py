import numpy as np
from sklearn.covariance import LedoitWolf

def ledoit_wolf_correlation(time_series_list, **kwargs):
    """
    Calcula a matriz de correlação usando o estimador Ledoit-Wolf para cada paciente.
    Retorna uma lista com os elementos do triângulo superior de cada matriz.
    """
    results = []
    for ts in time_series_list:
        data = ts.to_numpy()
        # Centraliza os dados (média zero por coluna)
        data = data - data.mean(axis=0)
        lw = LedoitWolf().fit(data)
        cov = lw.covariance_
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        corr = np.clip(corr, -1, 1)
        iu = np.triu_indices(corr.shape[0])
        results.append(corr[iu])
    return results
