#%%
#%%
from comet_ml import Experiment, OfflineExperiment
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
import skopt

def testing_metrics(y_test: np.ndarray, y_pred: np.ndarray):
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)
    
    true_positives = np.sum(y_test * y_pred)
    false_positives = np.sum(np.abs(y_test - 1) * y_pred)
    true_negatives = np.sum((y_test - 1) * (y_pred - 1))
    false_negatives = np.sum(y_test * np.abs(y_pred - 1))
    
    accuracy = round(
        (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives), 4)
    precision = round(true_positives / (true_positives + false_positives), 4)
    recall = round(true_positives / (true_positives + false_negatives), 4)
    specificity = round(true_negatives / (true_negatives + false_positives), 4)
    npv = round(true_negatives / (true_negatives + false_negatives), 4)
    f1_1 = round(2 * (precision * recall) / (precision + recall), 4)
    f1_0 = round(2 * (specificity * npv) / (specificity + npv), 4)
    f1_macro = round((f1_1 + f1_0) / 2, 4)
    
    return dict(
        accuracy = accuracy, f1_macro = f1_macro,
        f1_1 = f1_1, f1_0 = f1_0,
        Precision = precision, Recall = recall,
        Specificity = specificity, npv = npv,
        TP = int(true_positives), FP = int(false_positives), FN = int(false_negatives),
        TN = int(true_negatives),
    )

ALGORITHM = 'KNN'
DS = 'DS2'

#%%

for SEGMENTS_LENGTH in [4]:

    EXPERIMENT_ID = F'BayesSearch_{ALGORITHM}_{DS}_{SEGMENTS_LENGTH}s'

    data_dir = f'data\MODELS_DATA\\afdb\DS1'
    X_train = pd.read_csv(f'{data_dir}\\segments_{SEGMENTS_LENGTH}s_train.csv')
    X_test = pd.read_csv(f'{data_dir}\\segments_{SEGMENTS_LENGTH}s_test.csv')

    y_train = X_train.pop('episode')
    y_test = X_test.pop('episode')

    results = list()
    workflow_dict = dict()

    hyperparameters_optimizer = BayesSearchCV(
        KNeighborsClassifier(),
        {
            'n_neighbors': (2, 50),
            'weights': ['distance', 'uniform'],
            'p': (1,5),
            'metric': ['minkowski'],
            'n_jobs': [2],
        },
        n_iter=100,
        cv=2,
        verbose = 10,
        n_jobs = 2,
        n_points = 2,
        scoring = 'accuracy',
        random_state = 42
    )

    checkpoint_callback = skopt.callbacks.CheckpointSaver(f'D:\\FINKI\\8_dps\\Project\\MODELS\\afdb\\skopt_checkpoints\\{EXPERIMENT_ID}.pkl')
    hyperparameters_optimizer.fit(X_train, y_train, callback = [checkpoint_callback])
    skopt.dump(hyperparameters_optimizer, f'saved_models\\{EXPERIMENT_ID}.pkl')

    y_pred = hyperparameters_optimizer.best_estimator_.predict(X_test)

    exp = Experiment(
        api_key = 'A8Lg71j9LtIrsv0deBA0DVGcR',
        project_name = 'best-models-afib',
        workspace = "8_dps",
        auto_output_logging = 'native',
    )
    exp.set_name(f'{EXPERIMENT_ID}')
    exp.add_tags([DS, SEGMENTS_LENGTH, ALGORITHM])
    exp.log_parameters(hyperparameters_optimizer.best_estimator_.get_params())
    exp.log_metrics(testing_metrics(y_test = y_test, y_pred = y_pred))
    exp.end()

    for i in range(len(hyperparameters_optimizer.cv_results_['params'])):
        exp = Experiment(
                api_key = 'A8Lg71j9LtIrsv0deBA0DVGcR',
                project_name = f'{ALGORITHM}-afib',
                workspace = "8_dps",
                auto_output_logging = 'native',
            )
        exp.set_name(f'{EXPERIMENT_ID}_{i+1}')
        exp.add_tags([DS, SEGMENTS_LENGTH])
        for k, v in hyperparameters_optimizer.cv_results_.items():
            if k == "params": exp.log_parameters(dict(v[i]))
            else: exp.log_metric(k, v[i])
        exp.end()



#%%


