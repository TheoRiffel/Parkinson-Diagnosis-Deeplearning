import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso

def graphical_lasso_correlation(time_series_list: list[pd.DataFrame], alpha: float = 0.5, return_upper_triangular: bool = True) -> list[np.array]:
    """
    Computa a matriz de covariância esparsa penalizada com l1
    """


    correlation_matrices = []
    channels = time_series_list[0].shape[1]
    triu_indices = [np.triu_indices(channels)]
    for ts in time_series_list:
        X = ts.to_numpy()
        # Padroniza (não remove colunas)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        model = GraphicalLasso(alpha=alpha, max_iter=200)
        try:
            model.fit(X)
            precision = model.precision_
            d = np.sqrt(np.diag(precision))
            corr = precision / np.outer(d, d)
            corr = (corr + corr.T) / 2
            if return_upper_triangular:
                correlation_matrices.append(corr[triu_indices])
            else:
                correlation_matrices.append(corr)
        except FloatingPointError:
            n = X.shape[1]
            if return_upper_triangular:
                correlation_matrices.append(np.zeros(int(n * (n + 1) / 2)))
            else:
                correlation_matrices.append(np.zeros((channels, channels)))
    return correlation_matrices