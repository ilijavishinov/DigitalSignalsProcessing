#%%
from comet_ml import Experiment, OfflineExperiment
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
import testing_utils
import random

search_space_dir = 'search_spaces'

ALGORITHM = 'ANN'
DS = 'DS2'
SEGMENTS_LENGTH = 10
NUM_ITER = 100
#%%

EXPERIMENT_ID = F'RandomSearch_{ALGORITHM}_{DS}_{SEGMENTS_LENGTH}s'

data_dir = f'data\MODELS_DATA\\afdb\DS1'
X_train = pd.read_csv(f'{data_dir}\\segments_{SEGMENTS_LENGTH}s_train.csv')
X_test = pd.read_csv(f'{data_dir}\\segments_{SEGMENTS_LENGTH}s_test.csv')

y_train = X_train.pop('episode')
y_test = X_test.pop('episode')

#%%

import itertools

search_space = dict(
    model_architecture = [1,2,3],
    optimizer = ['adam', 'ftrl', 'rmsprop'],
    loss = ['categorical_hinge', 'binary_crossentropy', 'poisson', 'kl_divergence'],
    initializer = ['variance_scaling', 'glorot_normal'],
    activation = ['relu', 'sigmoid']
)

# combinations
keys, values = zip(*search_space.items())
combinations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

model_names = [f'{EXPERIMENT_ID}_{i + 1}' for i in range(len(combinations_dicts))]
combinations_df = pd.DataFrame(combinations_dicts)
combinations_df.insert(0, 'Model', model_names)
combinations_df['Trained'] = 'No'

search_space_path = f'{search_space_dir}\\search_space_{EXPERIMENT_ID}.xlsx'
combinations_df.to_excel(search_space_path, index = False)

more_models_left_to_train = True
results_list = list()


while NUM_ITER > 0:
    NUM_ITER -= 1

    search_space_state = pd.read_excel(search_space_path)

    random_index = random.choice(list(search_space_state.loc[search_space_state['Trained'] == 'No'].index))
    search_space_model = search_space_state.loc[search_space_state['Trained'] == 'No'].loc[random_index]

    comet_experiment = Experiment(
        api_key = 'A8Lg71j9LtIrsv0deBA0DVGcR',
        project_name = f'{ALGORITHM}-afib',
        workspace = "8_dps",
        auto_output_logging = 'native',
    )
    comet_experiment.set_name(search_space_model['Model'])
    comet_experiment.add_tags([DS, SEGMENTS_LENGTH, ALGORITHM])

    if search_space_model['model_architecture'] == 1:
        model = keras.Sequential([
            Dense(units = int(X_train.shape[1]/2), input_shape = (X_train.shape[1], ),
                  activation = search_space_model['activation'],
                  kernel_initializer = search_space_model['initializer']),
            Dense(units = 1, activation = 'sigmoid')
        ])

    if search_space_model['model_architecture'] == 2:
        model = keras.Sequential([
            Dense(units = int(2 * X_train.shape[1]/3), input_shape = (X_train.shape[1], ),
                  activation = search_space_model['activation'],
                  kernel_initializer = search_space_model['initializer']),
            Dense(units = int(X_train.shape[1]/3),
                  activation = search_space_model['activation'],
                  kernel_initializer = search_space_model['initializer']),
            Dense(units = 1, activation = 'sigmoid')
        ])

    if search_space_model['model_architecture'] == 3:
        model = keras.Sequential([
            Dense(units = int(3 * X_train.shape[1]/4), input_shape = (X_train.shape[1], ),
                  activation = search_space_model['activation'],
                  kernel_initializer = search_space_model['initializer']),
            Dense(units = int(X_train.shape[1]/2),
                  activation = search_space_model['activation'],
                  kernel_initializer = search_space_model['initializer']),
            Dense(units = int(3 * X_train.shape[1]/2),
                  activation = search_space_model['activation'],
                  kernel_initializer = search_space_model['initializer']),
            Dense(units = 1, activation = 'sigmoid')
        ])

    model.compile(loss = search_space_model['loss'],
                  optimizer = search_space_model['optimizer'],
                  metrics= ["accuracy"])

    with comet_experiment.train():

        model.fit(
            x = X_train,
            y = y_train,
            epochs = 1500,
            batch_size = 1000,
            validation_split = 0.1,
            callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 10, patience = 100)],
            verbose = 10,
            workers = 2
        )

    with comet_experiment.test():
        loss, accuracy = model.evaluate(X_test, y_test, verbose = 10, batch_size = 1000000, workers = 2)

    y_pred = model.predict_classes(X_test)
    comet_experiment.log_metrics(testing_utils.testing_metrics(y_test = y_test, y_pred = y_pred))

    model.save(f'saved_models/{search_space_model["Model"]}.h5')

    comet_experiment.end()

    search_space_state = pd.read_excel(search_space_path)
    search_space_state.loc[search_space_state['Model'] == search_space_model['Model'], 'Trained'] = 'Yes'
    search_space_state.to_excel(search_space_path, index = False)




#%%


