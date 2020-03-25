
from hyperopt import Trials, STATUS_OK, tpe, rand
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras import optimizers
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def get_data():
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

# split a univariate dataset into train/test sets
def train_test_split(data, index_test,index_val):
	return data[:-index_test], data[-index_test:], data[-index_val:]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return np.sqrt(mean_squared_error(actual, predicted))

# fit a model
def model_fit(model,train, val):
       
    # prepare data
    n_diff=2
    if n_diff > 0:
        train = difference(train, n_diff)
        val= difference(val, n_diff)
    
    # transform series into supervised format
    data_train = series_to_supervised(train, n_in={{choice([1, 5, 15, 20])}})
    data_val = series_to_supervised(val, n_in={{choice([1, 5, 15, 20])}})
    
    # separate inputs and outputs
    train_x, train_y = data_train[:, :-1], data_train[:, -1]
    val_x, val_y = data_val[:, :-1], data_val[:, -1]
    
    from keras import callbacks
    
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(train_x,
              train_y,
              epochs={{choice([25, 50, 75, 100])}},
              batch_size={{choice([16, 32, 64])}},
              validation_data=(val_x, val_y),
              callbacks=[reduce_lr])
    return model

# forecast with the fit model
def model_predict(model, history):
    
    # prepare data
    correction = 0.0
    n_diff=2
    if n_diff > 0:
        correction = history[-n_diff]
        history = difference(history, n_diff)
	# shape input for model
    x_input = array(history[-{{choice([1, 5, 15, 20])}}:]).reshape((1, {{choice([1, 5, 15, 20])}}))
	# make forecast
    yhat = model.predict(x_input, verbose=0)
	# correct forecast if it was differenced
    return correction + yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data,model, n_test, n_val):
    
    predictions = list()
    
    # split dataset
    index_test = int(len(data)*1- n_test )
    index_val = int(len(data)*1- n_val )
    train, test, val = train_test_split(data, index_test,index_val)
    # fit model
    model = model_fit(model,train,val)
	# seed history with training dataset
    history = [x for x in train]
	# step over each time-step in the test set
    for i in range(len(test)):
		# fit model and make forecast for history
        yhat = model_predict(model, history)
		# store forecast in list of predictions
        predictions.append(yhat)
		# add actual observation to history for the next loop
        history.append(test[i])
	# estimate prediction error
    error = measure_rmse(test, predictions)
    print(' > %.3f' % error)
    
    return error

# score a model, return None on failure
def repeat_evaluate(data,model, n_repeats=10):

    n_test = 0.20
    n_val = 0.25
	
	# fit and evaluate the model n times
    scores = [walk_forward_validation(data,model, n_test,n_val ) for _ in range(n_repeats)]
	# summarize score
    result = np.mean(scores)
    print('> Result %.3f' % result)
    return result

def create_model(data):

    model = Sequential()
    model.add(Dense({{choice([np.power(2, 5), np.power(2, 6), np.power(2, 7)])}}, input_shape= ({{choice([1, 5, 15, 20])}},)))
    model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense({{choice([np.power(2, 3), np.power(2, 4), np.power(2, 5)])}}))
    model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss='mse',
                  metrics=['accuracy'])
   
    repeat_evaluate(data,model,10)

    
    index_val = int(len(data)*1- 0.25 )
    index_test= 0
    train, test, val = train_test_split(data, index_test,index_val)

    # prepare data
    n_diff=2
    if n_diff > 0:        
        val= difference(train, n_diff)

    # transform series into supervised format    
    data_val = series_to_supervised(val, n_in={{choice([1, 5, 15, 20])}})
    
    # separate inputs and outputs    
    val_x, val_y = data_val[:, :-1], data_val[:, -1]

    score, acc = model.evaluate(val_x, val_y, verbose=0)
    print('Test accuracy:', acc)

    return {'loss': -10, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=create_model,
                                          data=get_data,
                                          algo=tpe.suggest,
                                          max_evals=15,
                                          trials=Trials())
   
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    best_model.save('breast_cancer_model.h5')
