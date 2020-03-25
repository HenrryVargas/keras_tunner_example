import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

import time
import pandas as pd
from loguru import logger
from pathlib import Path


from keras_t_data import load_data_raw,difference,series_to_supervised
from keras_t_class import define_tuners,NNTSMLPModel,MyTunner

from kerastuner.tuners import BayesianOptimization
from kerastuner import Objective

output_dir = Path("C://output/nn/")
tensorboard = TensorBoard(log_dir=output_dir)

def set_gpu_config():
    # Set up GPU config
    logger.info("Setting up GPU if found")
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

def tuner_evaluation(tuner, train_ds):
    
    # Overview of the task
    tuner.search_space_summary()

    # Performs the hyperparameter tuning
    logger.info("Start hyperparameter tuning")
    search_start = time.time()
    tuner.search(train_ds, callbacks=[tensorboard],validation_split=0.1)
    search_end = time.time()
    elapsed_time = search_end - search_start

    # Show a summary of the search
    tuner.results_summary()

    # Retrieve the best model.
    best_hps = tuner.get_best_hyperparameters()[0]
    print('Best hyperparameters')
    print(best_hps.values)
    best_model = tuner.get_best_model()[0]
   

def run_hyperparameter_tuning():
    hypermodel = NNTSMLPModel(5,1)
    
    tuner = MyTunner(BayesianOptimization(hypermodel= hypermodel,
                                          objective= Objective('loss','min'),                                          
                                          max_trials=2)
                    )
    # BayesianOptimization(
    #     hypermodel,
    #     objective='val_loss',
    #     seed=SEED,
    #     num_initial_points=BAYESIAN_NUM_INITIAL_POINTS,

    data_raw= load_data_raw()     
    # transform series into supervised format
    data_train = series_to_supervised(data_raw, n_in=5)
    tuner_evaluation(tuner,data_train)             
       
    
if __name__ == "__main__":
    # set_gpu_config()    
    run_hyperparameter_tuning()