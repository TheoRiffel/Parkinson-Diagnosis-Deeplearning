import os

import pandas as pd
import networkx as nx
import numpy as np

from fastdtw import fastdtw
from tqdm import tqdm
from node2vec import Node2Vec

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from scipy.spatial.distance import euclidean

def batch_read(path: str) -> list[pd.DataFrame]:
    df_list = []
    for file in tqdm(os.listdir(path)):
        df = pd.read_csv(f'{path}/{file}')
        df_list.append(df)
    return df_list

def dtw_distance(time_series1: np.array, time_series2: np.array) -> float:
    distance, _ = fastdtw(time_series1.reshape(-1, 1), time_series2.reshape(-1, 1), dist=euclidean)
    return distance

def compute_functional_network(time_series: np.array, distance_function: callable) -> np.array:
    n = time_series.shape[0]
    functional_network = np.zeros((n, n))
    for i in tqdm(range(1, n), leave=True):
        for j in tqdm(range(i+1, n), leave=False):
            distance = distance_function(time_series[i].reshape(-1, 1), time_series[j].reshape(-1, 1))
            functional_network[i, j] = distance
            functional_network[j, i] = distance
    return functional_network

if __name__ == '__main__':
    control_data = batch_read('data/Controls_columns')
    parkinson_data = batch_read('data/PDs_columns')

    np.unique([column.split('.')[0] for column in parkinson_data[0].columns])

    all_columns = parkinson_data[0].columns
    AAL3_columns = all_columns[np.where([column.split('.')[0] == 'AAL3' for column in all_columns])[0]]
    AAL3_columns[:5]

    control_AAL3_data = [data[AAL3_columns] for data in control_data]
    parkinson_AAL3_data = [data[AAL3_columns] for data in parkinson_data]

    compute_functional_network(parkinson_AAL3_data[0].to_numpy(), dtw_distance)

    parkinson_AAL3_data[0].shape

    upper_triangular_indices = np.triu_indices(166)

    parkinson_correlation_matrix = [time_series.corr().to_numpy()[upper_triangular_indices] for time_series in parkinson_AAL3_data]
    control_correlation_matrix = [time_series.corr().to_numpy()[upper_triangular_indices] for time_series in control_AAL3_data]

    X = np.concatenate([
        parkinson_correlation_matrix,
        control_correlation_matrix
    ], axis=0)

    y = np.concatenate([
        [1 for _ in range(len(parkinson_data))],
        [0 for _ in range(len(control_data))]
    ])

    notna_indices = np.logical_not(np.isnan(X).any(axis=1))
    X = X[notna_indices, :]
    y = y[notna_indices]
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    params = {
        'alpha': np.arange(1e-4, 1e-2, 1e-3),
        'learning_rate': ['constant', 'adaptive'],
    }

    model = MLPClassifier(early_stopping=True, hidden_layer_sizes=(100, 100))
    hyperparam_optimization = RandomizedSearchCV(model, params, random_state=1)
    search = hyperparam_optimization.fit(X_train, y_train)
    search.best_params_

    optimized_model = MLPClassifier(early_stopping=True, **search.best_params_).fit(X_train, y_train)
    optimized_model.score(X_test, y_test)

    optimized_model.predict(X_train)