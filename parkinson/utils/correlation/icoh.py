import os
import sys
import time

import numpy as np
import pandas as pd

from pathlib import Path
from joblib import Parallel, delayed
from scipy.signal import csd, welch

from .utils import save_cache, load_cache, min_max_normalize

def compute_icoh_matrix(time_series, sfreq=1000, fmin=8, fmax=30, nperseg=256):
    data = time_series.to_numpy().T  # shape (n_canais, n_amostras)
    n_channels = data.shape[0]
    n_samples = data.shape[1]
    nperseg = min(nperseg, n_samples)
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

    for batch_start in range(0, len(indices_to_process), n_jobs):
        batch_indices = indices_to_process[batch_start:batch_start + n_jobs]
        print(f"\nProcessando batch {batch_start} a {batch_start + len(batch_indices) - 1}...")

        new_results = Parallel(n_jobs=n_jobs)(
            delayed(process_and_save)(i) for i in batch_indices
        )

        for idx, res in new_results:
            completed_results[idx] = res

        save_cache(cache_path, completed_results)
        print(f"Cache parcial salvo após batch {batch_start}.")

    ordered_results = [completed_results[i] for i in range(total)]
    return ordered_results

def icoh_correlation(time_series_list: list[pd.DataFrame], group: str, n_jobs: int=6) -> list[np.array]:
    """
    Computa a correlação de Imaginary Coherence, que considera a parcela imaginária da 
    correlação entre séries no domínio das frequências.

    :param str group: 'parkinson' ou 'control'
    :return: matriz de correlação iCOH por paciente
    """
    root = Path(__file__).resolve().parents[3]
    sys.path.append('/')
    cache_dir = os.path.join(root, 'data/iCOH_matrix')
    os.makedirs(cache_dir, exist_ok=True)
    if group == 'parkinson':
        cache_path = os.path.join(cache_dir, 'cache_icoh_parkinson_final.pkl')
    elif group == 'control':
        cache_path = os.path.join(cache_dir, 'cache_icoh_control_final.pkl')
    else:
        raise ValueError("group deve ser 'parkinson' ou 'control'")
    icoh_matrices = compute_all_icoh_matrices(time_series_list, cache_path, n_jobs=n_jobs)
    return [min_max_normalize(mat) for mat in icoh_matrices]