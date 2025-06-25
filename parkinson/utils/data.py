from typing import Optional
import torch
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

avaiable_atlas = ['Shen_268', 'atlas', 'AAL3']

def batch_read(path: str) -> list[pd.DataFrame]:
    """Read .csv timeseries from folder"""
    df_list = []
    for file in tqdm(os.listdir(path)):
        df = pd.read_csv(f'{path}/{file}')
        df_list.append(df)
    return df_list

def select_atlas_columns(
        data: list[pd.DataFrame],
        atlas_name: str
    ) -> list[pd.DataFrame]:
    """Select atlas columns from df"""

    if atlas_name not in avaiable_atlas:
        raise ValueError(f'Invalid atlas name {atlas_name}. Use {avaiable_atlas}')
    
    all_columns = data[0].columns
    selected_columns = all_columns[np.where([column.split('.')[0] == atlas_name for column in all_columns])[0]]

    selected_data = [df[selected_columns] for df in data]

    return selected_data

def concatenate_data(
        data1: list[np.array],
        data2: list[np.array],
        data3: list[np.array] = None
    ) -> np.array:
    """
    Stack two lists[np.array]
    """
    
    data = np.concatenate([
        data1,
        data2,
    ], axis=0)

    if data3 is not None:
        data = np.concatenate([data, data3], axis=0)

    return data

def filter_data(X: np.array, y: np.array) -> tuple[np.array, np.array]:
    """
    Substitui NaNs por 0 nas séries
    """
    X = np.nan_to_num(X, nan=0.0)
    return X, y

def get_torch_class_weights(y: np.ndarray) -> torch.Tensor:
    classes = np.unique(y)
    class_counts = np.array([(y == c).sum() for c in classes])
    weights = 1. / class_counts
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float32)

def get_torch_dataloader(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
    """"
    Get torch dataloader
    """
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)

    return loader

def df_to_timeseries(data_df):
    aux_list = []

    for idx_person in range(len(data_df)):
        person_ts = []

        for idx_region in range(len(data_df[idx_person].columns)):
            person_ts.append(list(data_df[idx_person].iloc[:,idx_region]))

        aux_list.append(person_ts)

    return aux_list