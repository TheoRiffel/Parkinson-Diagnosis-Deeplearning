import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import os

def batch_read(path: str) -> list[pd.DataFrame]:
    df_list = []
    for file in tqdm(os.listdir(path)):
        df = pd.read_csv(f'{path}/{file}')
        df_list.append(df)
    return df_list

def select_atlas_type(parkinson_data: list[pd.DataFrame],
                    control_data: list[pd.DataFrame],
                    atlas_name: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    
    all_columns = parkinson_data[0].columns
    selected_columns = all_columns[np.where([column.split('.')[0] == atlas_name for column in all_columns])[0]]

    parkinson_selected_data = [data[selected_columns] for data in parkinson_data]
    control_selected_data = [data[selected_columns] for data in control_data]

    return parkinson_selected_data, control_selected_data

def join_data(parkinson_correlation_matrix: list[np.ndarray],
              control_correlation_matrix: list[np.ndarray],
              len_park_data: int,
              len_control_data: int) -> tuple[np.ndarray, np.ndarray]:
    
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

def get_class_weights(y_train: np.ndarray) -> torch.Tensor:

    classes = np.unique(y_train)
    class_counts = np.array([(y_train == c).sum() for c in classes])
    weights = 1. / class_counts
    weights = weights / weights.sum() * len(classes)

    return torch.tensor(weights, dtype=torch.float32)

def load_dataloader_for_training(X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=16)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=16)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=16)

    return train_loader, val_loader, test_loader