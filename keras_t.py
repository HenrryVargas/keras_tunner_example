from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch,Hyperband,BayesianOptimization
import pandas as pd
from loguru import logger
import time
from pathlib import Path
from sklearn import preprocessing


n_input = [1, 5, 15, 20]
n_nodes = [50, 100]
n_epochs = [100]
n_batch = [1, 150]
n_diff = [0, 5, 15, 20]

HYPERBAND_MAX_EPOCHS = 40
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 2
N_EPOCH_SEARCH = 40
BAYESIAN_NUM_INITIAL_POINTS = 1
SEED=1
output_dir = Path("C://output/nn/")
tensorboard = TensorBoard(log_dir=output_dir)


def set_gpu_config():
    # Set up GPU config
    logger.info("Setting up GPU if found")
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

def load_data_raw():
    # define dataset
    series = pd.read_csv('createdDateEnglish.csv', sep=";", header=0, index_col=0)
    data = series.values
    return data   
     
# difference dataset
def difference(data, order):
	return [data[i] - data[i - order] for i in range(order, len(data))]

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values

class NNTSMLPModel(HyperModel):
    def __init__(self, input_shape=None, num_classes=None):
       # if input_shape == 0:
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build(self, hp):       
        
        model =keras.models.Sequential()
        model.add(Dense(units=hp.Int(
                                    'units',
                                    min_value= 50,
                                    max_value=200,
                                    step=50,
                                    default=50
                                    ), 
                        activation=hp.Choice(
                                    'dense_activation',
                                    values=['relu', 'tanh', 'sigmoid'],
                                    default='relu'
                                    ),
                        input_dim= self.input_shape
                        )
                )
        for i in range(hp.Int('num_layers', 1, 6)):
                                            model.add(keras.layers.Dense(
                                                units=hp.Int('units_' + str(i),
                                                            min_value=5,
                                                            max_value=200,
                                                            step=20),
                                             
                                            activation=hp.Choice(
                                                        'dense_activation',
                                                        values=['relu', 'tanh', 'sigmoid'],
                                                        default='relu'
                                                        ),
                                            ))

                                            model.add(Dropout(rate=hp.Float(
                                                                            'dropout'+ str(i),
                                                                            min_value=0.0,
                                                                            max_value=0.5,
                                                                            default=0.25,
                                                                            step=0.05
                                                                        )
                                                            )
                                            )
        
        model.add(Dense(self.num_classes ,activation=hp.Choice('dense_activation_final',
                                                values=['relu', 'tanh', 'sigmoid'],
                                                default='sigmoid'
                                              )
                        )
                )
        
        
        model.compile(loss='mape', metrics=['MeanAbsoluteError'], optimizer=keras.optimizers.Adam(hp.Float(
                                                                            'learning_rate',
                                                                            min_value=1e-4,
                                                                            max_value=1e-2,
                                                                            sampling='LOG',
                                                                            default=1e-3
                                                                        ),
                                                                        
                                                                 ),
                     )
        return model

def define_tuners(hypermodel, directory, project_name):
    random_tuner = RandomSearch(
        hypermodel,
        objective="val_loss",
        seed=SEED,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory=f"{directory}_random_search",
        project_name=project_name,
    )
    hyperband_tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        objective="val_loss",
        seed=SEED,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory=f"{directory}_hyperband",
        project_name=project_name,
    )
    bayesian_tuner = BayesianOptimization(
        hypermodel,
        objective='val_loss',
        seed=SEED,
        num_initial_points=BAYESIAN_NUM_INITIAL_POINTS,
        max_trials=MAX_TRIALS,
        directory=f"{directory}_bayesian",
        project_name=project_name
    )
    return [random_tuner, hyperband_tuner, bayesian_tuner]

def tuner_evaluation(tuner, x_test, x_train, y_test, y_train):
    #set_gpu_config()

    # Overview of the task
    tuner.search_space_summary()

    # Performs the hyperparameter tuning
    logger.info("Start hyperparameter tuning")
    search_start = time.time()
    tuner.search(x_train, y_train,  callbacks=[tensorboard], epochs=30, validation_split=0.1)
    search_end = time.time()
    elapsed_time = search_end - search_start

    # Show a summary of the search
    tuner.results_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(x_test, y_test)
    return elapsed_time, loss, accuracy


def run_hyperparameter_tuning():
    values= load_data_raw()     
    #0,5, 15, 20
    for i in [5]:
        if i > 0:
            values = difference(values, i)
        
        # transform series into supervised format
        data_train = series_to_supervised(values, n_in=i)
        data_test = series_to_supervised(values, n_in=i)
        
        # separate inputs and outputs
        x_train, y_train = data_train[:, :-1], data_train[:, -1]        
        x_test, y_test = data_test[:, :-1], data_test[:, -1]       
    
        hypermodel = NNTSMLPModel(i,1)
        tuners = define_tuners(hypermodel, directory=output_dir, project_name="simple_mlp_tuning")
       
        results = []
        for tuner in tuners:
            elapsed_time, loss, accuracy = tuner_evaluation(tuner, x_test, x_train, y_test, y_train)
            logger.info(f"Elapsed time = {elapsed_time:10.4f} s, accuracy = {accuracy}, loss = {loss}")
            results.append([elapsed_time, loss, accuracy])
        
        logger.info(results)

if __name__ == "__main__":
    
    run_hyperparameter_tuning()