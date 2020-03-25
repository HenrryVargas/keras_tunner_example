import pandas as pd
from sklearn import preprocessing
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
