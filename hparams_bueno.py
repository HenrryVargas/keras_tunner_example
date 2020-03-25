from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd


import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("GPU Available: ", tf.test.is_gpu_available())

data = pd.read_csv('train.csv')

# Missing value handling
mean_value = round(data['Age'].mean())
mode_value = data['Embarked'].mode()[0]

value = {'Age': mean_value, 'Embarked': mode_value}
data.fillna(value=value,inplace=True)

data.dropna(axis=1,inplace=True)

#Train, val, test Split
train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.25)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


#input pipeline using tf.data

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Survived')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


batch_size = 32 
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)



#Feature columns

#numarical features
num_c = ['Age','Fare','Parch','SibSp'] 
bucket_c  = ['Age'] #bucketized numerical feature
#categorical features
cat_i_c = ['Embarked', 'Pclass','Sex'] #indicator columns
cat_e_c = ['Ticket'] # embedding column


def get_scal(feature):
  def minmax(x):
    mini = train[feature].min()
    maxi = train[feature].max()
    return (x - mini)/(maxi-mini)
  return(minmax)

# Create Feature Columns

# Numerical columns
feature_columns = []
for header in num_c:
  scal_input_fn = get_scal(header)
  feature_columns.append(feature_column.numeric_column(header, normalizer_fn=scal_input_fn))

# Bucketized columns
Age = feature_column.numeric_column("Age")
age_buckets = feature_column.bucketized_column(Age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# Categorical indicator columns
for feature_name in cat_i_c:
  vocabulary = data[feature_name].unique()
  cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
  one_hot = feature_column.indicator_column(cat_c)
  feature_columns.append(one_hot)

# Categorical embedding columns
for feature_name in cat_e_c:
  vocabulary = data[feature_name].unique()
  cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
  embeding = feature_column.embedding_column(cat_c, dimension=50)
  feature_columns.append(embeding)

# Crossed columns
vocabulary = data['Sex'].unique()
Sex = tf.feature_column.categorical_column_with_vocabulary_list('Sex', vocabulary)

crossed_feature = feature_column.crossed_column([age_buckets, Sex], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)
print(len(feature_columns))


HP_NUM_UNITS1 = hp.HParam('num_units 1', hp.Discrete([50,100,150])) 
HP_NUM_UNITS2 = hp.HParam('num_units 2', hp.Discrete([30,50]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.2))
HP_DROPOUT2 = hp.HParam('dropout2', hp.RealInterval(0.3,0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','RMSprop']))
HP_L2 = hp.HParam('l2 regularizer', hp.RealInterval(.001,.01))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS1,HP_NUM_UNITS2, HP_DROPOUT,HP_L2 ,HP_OPTIMIZER,HP_DROPOUT2],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

feature_layer = tf.keras.layers.DenseFeatures (feature_columns)

def train_test_model(hparams):
  model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(hparams[HP_NUM_UNITS1], kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    layers.Dropout(hparams[HP_DROPOUT]),
    layers.Dense(hparams[HP_NUM_UNITS2], kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),    
    layers.Dropout(hparams[HP_DROPOUT2]),
    layers.Dense(1, activation='sigmoid')
  ])
  
  model.compile(optimizer=hparams[HP_OPTIMIZER],
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.fit(train_ds,validation_data=val_ds,epochs=10)
  _, accuracy = model.evaluate(val_ds)
  
  return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)



session_num = 0
for num_units1 in HP_NUM_UNITS1.domain.values:
  for num_units2 in HP_NUM_UNITS2.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
      for dropout_rate2 in (HP_DROPOUT2.domain.min_value, HP_DROPOUT2.domain.max_value):  
        for l2 in (HP_L2.domain.min_value, HP_L2.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:            
                hparams = {
                    HP_NUM_UNITS1: num_units1,
                    HP_NUM_UNITS2: num_units2,
                    HP_DROPOUT: dropout_rate,
                    HP_DROPOUT2: dropout_rate2,
                    HP_L2: l2,
                    HP_OPTIMIZER: optimizer
                    
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + run_name, hparams)
                session_num += 1



