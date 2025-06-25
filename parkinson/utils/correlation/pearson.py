import numpy as np
import pandas as pd

def pearson_correlation(time_series_list: list[pd.DataFrame]) -> list[np.array]:
    """
    Retorna a correlação de Pearson para a lista de séries temporais
    """
    return [ts.corr(method='pearson').to_numpy()[np.triu_indices(ts.shape[1])] for ts in time_series_list]