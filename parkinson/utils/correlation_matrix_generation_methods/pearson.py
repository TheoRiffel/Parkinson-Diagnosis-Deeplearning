import numpy as np

def pearson_correlation(time_series_list, **kwargs):
    return [ts.corr(method='pearson').to_numpy()[np.triu_indices(ts.shape[1])] for ts in time_series_list]