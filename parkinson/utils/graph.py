from .correlation_matrix_generation_methods.pearson import pearson_correlation
from .correlation_matrix_generation_methods.spearman import spearman_correlation
from .correlation_matrix_generation_methods.dtw import dtw_correlation
from .correlation_matrix_generation_methods.icoh import icoh_correlation
from .correlation_matrix_generation_methods.graphical_lasso import graphical_lasso_correlation
from .correlation_matrix_generation_methods.ledoit_wolf import ledoit_wolf_correlation
from .correlation_matrix_generation_methods.sliding_window import sliding_window_correlation

def compute_correlation_matrix(
    time_series_list,
    method='pearson',
    group='parkinson',
    n_jobs: int = 6
):
    """
    Seleciona o método de correlação e executa a função correspondente.
    """
    methods = {
        'pearson': pearson_correlation,
        'spearman': spearman_correlation,
        'dtw': dtw_correlation,
        'icoh': icoh_correlation,
        'graphical_lasso': graphical_lasso_correlation,
        'ledoit_wolf': ledoit_wolf_correlation,
        'sliding_window': sliding_window_correlation
    }
    if method not in methods:
        raise ValueError(f"Método de geração de matriz de correlação desconhecido: {method}")

    return methods[method](time_series_list, group=group, n_jobs=n_jobs)