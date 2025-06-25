import numpy as np
import pandas as pd

def spearman_correlation(time_series_list: list[pd.DataFrame]) -> list[np.array]:
    """
    Retorna a correlação de Spearman para a lista de séries temporais
    """
    return [ts.corr(method='spearman').to_numpy()[np.triu_indices(ts.shape[1])] for ts in time_series_list]