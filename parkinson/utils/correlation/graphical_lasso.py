import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso

def graphical_lasso_correlation(time_series_list: list[pd.DataFrame], alpha: float = 0.5) -> list[np.array]:
    """
    Computa a matriz de covariância esparsa penalizada com l1
    """
    correlation_matrices = []
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
            iu = np.triu_indices_from(corr)
            correlation_matrices.append(corr[iu])
        except FloatingPointError:
            n = X.shape[1]
            correlation_matrices.append(np.zeros(int(n * (n + 1) / 2)))
    return correlation_matrices