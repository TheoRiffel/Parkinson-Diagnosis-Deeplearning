import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw
from tqdm import tqdm

import pickle
import os
import sys
import time
from joblib import Parallel, delayed

# até onde eu sei essa função não tá sendo usada
def dtw_distance(time_series1: np.array, time_series2: np.array) -> float:
    distance, _ = fastdtw(time_series1.reshape(-1, 1), time_series2.reshape(-1, 1), dist=euclidean)
    return distance


def compute_functional_network(time_series: np.array, distance_function: callable) -> np.array:
    """
    Calcula a matriz de distâncias entre canais de uma série temporal usando a função de distância fornecida.
    time_series.shape == (n_canais, n_amostras)
    """
    n = time_series.shape[0]
    functional_network = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            distance = distance_function(time_series[i].reshape(-1, 1), time_series[j].reshape(-1, 1))
            functional_network[i, j] = distance
            functional_network[j, i] = distance
    return functional_network


# outra estimador para similaridade entre séries temporais (tipo um pearson)
def compute_dtw_matrix(time_series) -> np.array:
    """
    Calcula a matriz de distâncias DTW entre todos os pares de canais de uma série temporal.
    Retorna apenas os valores do triângulo superior (como na matriz de correlação).
    """
    # converte para array e coloca canais no eixo 0
    data = time_series.to_numpy().T  # shape (n_canais, n_amostras)
    # computa toda a matriz de distâncias
    dtw_net = compute_functional_network(data, dtw_distance)
    # extrai o triângulo superior (incluindo diagonal)
    iu = np.triu_indices(dtw_net.shape[0])
    return dtw_net[iu]


def save_cache(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_cache(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_all_dtw_matrices(time_series_list, cache_path, final_cache_path=None, n_jobs=4):
    if final_cache_path is None:
        final_cache_path = cache_path.replace('.pkl', '_final.pkl')

    # Carrega cache parcial existente, se houver
    if os.path.exists(cache_path):
        print(f"Carregando cache existente de {cache_path}...")
        completed_results = load_cache(cache_path)
    else:
        completed_results = {}

    total = len(time_series_list)
    indices_to_process = [i for i in range(total) if i not in completed_results]

    print(f"Total: {total} pacientes")
    print(f"Já computados: {len(completed_results)}")
    print(f"Restantes: {len(indices_to_process)}")

    def process_and_save(idx):
        print(f"Iniciando paciente {idx}...")
        start_time = time.time()
        ts = time_series_list[idx]
        result = compute_dtw_matrix(ts)
        elapsed = time.time() - start_time
        print(f"Paciente {idx} finalizado em {elapsed:.2f} segundos.")
        return idx, result

    # Processa os pacientes em batches de n_jobs
    for batch_start in range(0, len(indices_to_process), n_jobs):
        batch_indices = indices_to_process[batch_start:batch_start + n_jobs]

        print(f"\nProcessando batch {batch_start} a {batch_start + len(batch_indices) - 1}...")

        new_results = Parallel(n_jobs=n_jobs)(
            delayed(process_and_save)(i) for i in batch_indices
        )

        # Atualiza o cache com os resultados do batch
        for idx, res in new_results:
            completed_results[idx] = res

        # Salva o cache parcial após cada batch
        save_cache(cache_path, completed_results)
        print(f"Cache parcial salvo após batch {batch_start}.")

    # Reordena os resultados finais
    ordered_results = [completed_results[i] for i in range(total)]

    # Salva o resultado final completo
    save_cache(final_cache_path, completed_results)
    print(f"\nResultado final salvo em {final_cache_path}")

    return ordered_results


def compute_correlation_matrix(
    time_series_list,
    method='dtw',
    group='parkinson',  # 'parkinson' ou 'control'
):
    """
    Calcula a matriz de correlação para uma lista de séries temporais.
    method: 'pearson' ou 'dtw'
    group: 'parkinson' ou 'control' (usado para definir o cache_path do DTW)
    """
    if method == 'pearson':
        return [ts.corr(method='pearson').to_numpy()[np.triu_indices(ts.shape[1])] for ts in time_series_list]
    elif method == 'dtw':
        # Defina os parâmetros de configuração do DTW aqui:
        sys.path.append('/')
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        if group == 'parkinson':
            cache_path = os.path.join(cache_dir, 'cache_dtw_parkinson.pkl')
            n_jobs = 6
        elif group == 'control':
            cache_path = os.path.join(cache_dir, 'cache_dtw_control.pkl')
            n_jobs = 6
        else:
            raise ValueError("group deve ser 'parkinson' ou 'control'")
        return compute_all_dtw_matrices(time_series_list, cache_path, n_jobs=n_jobs)
    else:
        raise ValueError(f"Método desconhecido: {method}")