import numpy as np
from sklearn.model_selection import train_test_split
import torch
import time

from data_utils import batch_read, select_atlas_type, join_data, load_dataloader_for_training, get_class_weights
from train_utils import train, evaluate
from simpleMLP import SimpleMLP

RDN = 50
N_CLASSES = 2
BATCH_SIZE = 32

if __name__ == '__main__':

    #------------------------------ MANAGING DATA -----------------------------#
    parkinson_data = batch_read('data/PDs_columns')
    control_data = batch_read('data/Controls_columns')
    parkinson_selected_data, control_selected_data = select_atlas_type(parkinson_data, control_data, 'AAL3')

    upper_triangular_indices = np.triu_indices(166)
    parkinson_correlation_matrix = [time_series.corr().to_numpy()[upper_triangular_indices] for time_series in parkinson_selected_data]
    control_correlation_matrix = [time_series.corr().to_numpy()[upper_triangular_indices] for time_series in control_selected_data]

    X, y = join_data(parkinson_correlation_matrix, control_correlation_matrix, len(parkinson_data), len(control_data))

    # 60% treino, 20% validação, 20% teste
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=RDN, stratify=y, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=RDN, stratify=y_trainval, shuffle=True)
    train_loader, val_loader, test_loader = load_dataloader_for_training(X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE)

    #-------------------------------- TRAINING --------------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(input_dim=X_train.shape[1], hidden_dim=16, output_dim=2)
    class_weights = get_class_weights(y_train)

    start = time.time()
    out = train(model, train_loader, val_loader, class_weights, device)
    metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - start

    print('Metrics:', metrics)
    print('Time elapsed:', elapsed)
    print('Done.')