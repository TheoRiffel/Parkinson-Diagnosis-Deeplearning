import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_losses(train_loss: np.array, val_loss: np.array):
    """
    Plota losses de treino e teste
    """
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(train_loss)), train_loss, color='b', label='train')
    if val_loss != None:
        ax.plot(np.arange(len(val_loss)), val_loss, color='r', label='val')
    ax.legend()
    return fig

def plot_confusion_matrix(preds: np.array, labels: np.array, class_names: list[str] = None):
    """
    Plota a matriz de confusão
    """

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Matriz de Confusão")
    return fig

def metrics_to_dataframe(metrics: dict):
    data = {
        'Acurácia': [metrics['acc']],
        'F1': [metrics['f1']],
        'Precisão': [metrics['precision']],
        'Recall': [metrics['recall']]
    }

    print(pd.DataFrame(data).to_string(index=False))