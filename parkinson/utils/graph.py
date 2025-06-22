import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.signal import csd, welch

from fastdtw import fastdtw
from tqdm import tqdm
from pathlib import Path

import pickle
import os
import sys
import time
from joblib import Parallel, delayed

def save_cache(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_cache(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def min_max_normalize(arr):
    """
    Normaliza um array para o intervalo [0, 1].
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

# até onde eu sei essa função não tá sendo usada: não ta
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


def compute_all_dtw_matrices(time_series_list, cache_path, n_jobs=4):

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
        print(f"Iniciando paciente {idx+1}...")
        start_time = time.time()
        ts = time_series_list[idx]
        result = compute_dtw_matrix(ts)
        elapsed = time.time() - start_time
        print(f"Paciente {idx+1} finalizado em {elapsed:.2f} segundos.")
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

    return ordered_results


def compute_icoh_matrix(time_series, sfreq=1000, fmin=8, fmax=30, nperseg=256):
    """
    Calcula a matriz de iCOH entre todos os pares de canais de uma série temporal.
    """
    data = time_series.to_numpy().T  # shape (n_canais, n_amostras)
    n_channels = data.shape[0]
    n_samples = data.shape[1]
    nperseg = min(nperseg, n_samples)  # Garante que nperseg não seja maior que o número de amostras
    icoh_mat = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i, n_channels):
            f, Pxy = csd(data[i], data[j], fs=sfreq, nperseg=nperseg)
            _, Pxx = welch(data[i], fs=sfreq, nperseg=nperseg)
            _, Pyy = welch(data[j], fs=sfreq, nperseg=nperseg)
            coh_complex = Pxy / np.sqrt(Pxx * Pyy)
            freq_mask = (f >= fmin) & (f <= fmax)
            icoh_val = np.mean(np.abs(np.imag(coh_complex[freq_mask])))
            icoh_mat[i, j] = icoh_val
            icoh_mat[j, i] = icoh_val
    iu = np.triu_indices(n_channels)
    return icoh_mat[iu]


def compute_all_icoh_matrices(time_series_list, cache_path, n_jobs=4, sfreq=1000, fmin=8, fmax=30, nperseg=256):
    """
    Calcula as matrizes de iCOH para uma lista de séries temporais, com paralelização e cache.
    """

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
        print(f"Iniciando paciente {idx+1}...")
        start_time = time.time()
        ts = time_series_list[idx]
        result = compute_icoh_matrix(ts, sfreq=sfreq, fmin=fmin, fmax=fmax, nperseg=nperseg)
        elapsed = time.time() - start_time
        print(f"Paciente {idx+1} finalizado em {elapsed:.2f} segundos.")
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

    return ordered_results


def compute_correlation_matrix(
    time_series_list,
    method='dtw',
    group='parkinson',  # 'parkinson' ou 'control'
    n_jobs: int = 6
):
    """
    Calcula a matriz de correlação para uma lista de séries temporais.
    method: 'pearson' ou 'dtw'
    group: 'parkinson' ou 'control' (usado para definir o cache_path do DTW)
    """
    root = Path(__file__).parent.parent.parent

    if method == 'pearson':
        return [ts.corr(method='pearson').to_numpy()[np.triu_indices(ts.shape[1])] for ts in time_series_list]
    elif method == 'dtw':
        sys.path.append('/')
        cache_dir = os.path.join(root, 'data/dtw_matrix')
        os.makedirs(cache_dir, exist_ok=True)
        if group == 'parkinson':
            cache_path = os.path.join(cache_dir, 'cache_dtw_parkinson_final.pkl')
        elif group == 'control':
            cache_path = os.path.join(cache_dir, 'cache_dtw_control_final.pkl')
        else:
            raise ValueError("group deve ser 'parkinson' ou 'control'")
        dtw_matrices = compute_all_dtw_matrices(time_series_list, cache_path, n_jobs=n_jobs)
        return [min_max_normalize(mat) for mat in dtw_matrices]
    elif method == 'spearman':
        return [ts.corr(method='spearman').to_numpy()[np.triu_indices(ts.shape[1])] for ts in time_series_list]
    elif method == 'icoh':
        sys.path.append('/')
        cache_dir = os.path.join(root, 'data/iCOH_matrix')
        os.makedirs(cache_dir, exist_ok=True)
        if group == 'parkinson':
            cache_path = os.path.join(cache_dir, 'cache_icoh_parkinson_final.pkl')
            n_jobs = 6
        elif group == 'control':
            cache_path = os.path.join(cache_dir, 'cache_icoh_control_final.pkl')
            n_jobs = 6
        else:
            raise ValueError("group deve ser 'parkinson' ou 'control'")
        icoh_matrices = compute_all_icoh_matrices(time_series_list, cache_path, n_jobs=n_jobs)
        return [min_max_normalize(mat) for mat in icoh_matrices]
    else:
        raise ValueError(f"Método de geração de matriz de correlação desconhecido: {method}")