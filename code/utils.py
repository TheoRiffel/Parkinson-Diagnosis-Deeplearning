import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def batch_read(path: str) -> list[pd.DataFrame]:
    df_list = []
    for file in tqdm(os.listdir(path)):
        df = pd.read_csv(f'{path}/{file}')
        df_list.append(df)
    return df_list

def select_atlas_type(parkinson_data: list[pd.DataFrame], control_data: list[pd.DataFrame], atlas_name: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    all_columns = parkinson_data[0].columns
    selected_columns = all_columns[np.where([column.split('.')[0] == atlas_name for column in all_columns])[0]]

    parkinson_selected_data = [data[selected_columns] for data in parkinson_data]
    control_selected_data = [data[selected_columns] for data in control_data]

    return parkinson_selected_data, control_selected_data

def join_data(parkinson_correlation_matrix: list[np.ndarray], control_correlation_matrix: list[np.ndarray], len_park_data: int, len_control_data: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([
        parkinson_correlation_matrix,
        control_correlation_matrix
    ], axis=0)

    y = np.concatenate([
        [1 for _ in range(len_park_data)],
        [0 for _ in range(len_control_data)]
    ])

    notna_indices = np.logical_not(np.isnan(X).any(axis=1))
    X = X[notna_indices, :]
    y = y[notna_indices]

    return X, y