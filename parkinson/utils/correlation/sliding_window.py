import numpy as np
import pandas as pd

def get_window_size(time_series_list: list[pd.DataFrame], min_ratio: float, max_ratio: float) -> int:
    lengths = [ts.shape[0] for ts in time_series_list]
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = (min_len + max_len) / 2
    # Usando 40% como valor intermediário
    window_size = int(avg_len * 0.4)
    return max(2, window_size)  # pelo menos 2

def sliding_window_correlation(time_series_list: list[pd.DataFrame], min_ratio: float = 0.3, max_ratio: float = 0.5) -> list[np.array]:
    window_size = get_window_size(time_series_list, min_ratio, max_ratio)
    step_size = window_size // 2  # sobreposição de 50%
    results = []

    for ts in time_series_list:
        n_timepoints, n_regions = ts.shape
        matrices = []
        for start in range(0, n_timepoints - window_size + 1, step_size):
            window = ts.iloc[start:start+window_size]
            corr = window.corr(method='pearson').to_numpy()
            matrices.append(corr)
        if not matrices:
            # Se a série for menor que a janela, calcula a correlação de tudo
            corr = ts.corr(method='pearson').to_numpy()
            matrices.append(corr)
        mean_corr = np.mean(matrices, axis=0)
        iu = np.triu_indices(n_regions)
        results.append(mean_corr[iu])
    return results