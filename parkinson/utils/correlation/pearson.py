import numpy as np
import pandas as pd

def pearson_correlation(time_series_list: list[pd.DataFrame], return_upper_triangular: bool = True) -> list[np.array]:
    """
    Retorna a correlação de Pearson para a lista de séries temporais
    """
    correlation_matrices = [ts.corr(method='pearson').to_numpy() for ts in time_series_list]
    triu_indices = np.triu_indices(time_series_list[0].shape[1])
    if return_upper_triangular:
        return [correlation_matrix[triu_indices] for correlation_matrix in correlation_matrices]
    return correlation_matrices