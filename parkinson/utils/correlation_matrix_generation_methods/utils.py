import numpy as np
import pickle

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