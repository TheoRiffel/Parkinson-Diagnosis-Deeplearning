import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from utils import batch_read, select_atlas_type, join_data

RDN = 42

if __name__ == '__main__':

    ##### MANAGING DATA #####
    parkinson_data = batch_read('data/PDs_columns')
    control_data = batch_read('data/Controls_columns')
    parkinson_selected_data, control_selected_data = select_atlas_type(parkinson_data, control_data, 'AAL3')

    upper_triangular_indices = np.triu_indices(166)
    parkinson_correlation_matrix = [time_series.corr().to_numpy()[upper_triangular_indices] for time_series in parkinson_selected_data]
    control_correlation_matrix = [time_series.corr().to_numpy()[upper_triangular_indices] for time_series in control_selected_data]

    X, y = join_data(parkinson_correlation_matrix, control_correlation_matrix, len(parkinson_data), len(control_data))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=RDN,
        stratify=y
    )
    print(np.sum(y_test == 1) / len(y_test))
    
    ##### TRAINING #####
    params = {
        'alpha': np.arange(1e-4, 1e-2, 1e-3),
        'learning_rate': ['constant', 'adaptive'],
    }

    model = MLPClassifier(early_stopping=True, hidden_layer_sizes=(100, 100))
    hyperparam_optimization = RandomizedSearchCV(model, params, random_state=1)
    search = hyperparam_optimization.fit(X_train, y_train)
    search.best_params_

    optimized_model = MLPClassifier(early_stopping=True, **search.best_params_).fit(X_train, y_train)
    score = optimized_model.score(X_test, y_test)
    print(score)

    print(optimized_model.predict(X_test))