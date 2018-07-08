
# coding: utf-8

# ### Deep Neural Network Classifier of 2015 Flight Delay + Weather Data
# 
# Analysis performed with TensorFlow using a built-in deep learning estimator.


import numpy as np 
import pandas as pd
import tensorflow as tf

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

# Load weather and flight dataframe
df = pd.read_csv('data/FlightsAndWeatherData_top50Airports_NoMissingData.txt', sep='\t')
print('Dataframe dimensions:', df.shape)


# Summarize types and missing data
df_summary = pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
df_summary = df_summary.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
df_summary = df_summary.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)
                         .T.rename(index={0:'null values (%)'}))
df_summary


# Remove unnecessary columns
df = df.drop(['YEAR', 'DATE', 'SCHEDULED_ARRIVAL', 'SCHEDULED_DEPARTURE',
              'OriginDewPoint', 'OriginRelHumidity', 'OriginWindDirection',
              'OriginAltimeter', 'OriginWindGusts', 'ArrivalTemp','ArrivalDewPoint',
              'ArrivalRelHumidity', 'ArrivalWindDirection', 'ArrivalAltimeter', 
              'ArrivalVisibility', 'ORIGIN_CONGESTION', 'DESTINATION_CONGESTION'], 
             axis = 1)
print('Remaining columns: ', df.columns.tolist())
ncols = len(df.columns)


# Define prediction target
delayThreshold = 15
df['DELAYED'] = np.where(df['ARRIVAL_DELAY']> delayThreshold, 1, 0)
df = df.drop(['ARRIVAL_DELAY'], axis= 1)


# Build feature columns
my_columns = []
my_columns.append(tf.feature_column.numeric_column(key='MONTH'))
my_columns.append(tf.feature_column.numeric_column(key='DAY'))
my_columns.append(tf.feature_column.numeric_column(key='DAY_OF_WEEK'))
my_columns.append(tf.feature_column.embedding_column(categorical_column=
                                                     tf.feature_column.categorical_column_with_vocabulary_list(
                                                         key='AIRLINE',
                                                         vocabulary_list=sorted(pd.Series(df['AIRLINE']).unique())), 
                                                     dimension=3))
my_columns.append(tf.feature_column.embedding_column(categorical_column=
                                                     tf.feature_column.categorical_column_with_vocabulary_list(
                                                         key='ORIGIN_AIRPORT',
                                                         vocabulary_list=sorted(pd.Series(df['ORIGIN_AIRPORT']).unique())),
                                                     dimension=5))
my_columns.append(tf.feature_column.embedding_column(categorical_column=
                                                     tf.feature_column.categorical_column_with_vocabulary_list(
                                                         key='DESTINATION_AIRPORT',
                                                         vocabulary_list=sorted(pd.Series(df['DESTINATION_AIRPORT']).unique())),
                                                     dimension=5))
my_columns.append(tf.feature_column.numeric_column(key='DISTANCE'))
my_columns.append(tf.feature_column.numeric_column(key='OriginTemp'))
my_columns.append(tf.feature_column.numeric_column(key='OriginWindSpeed'))
my_columns.append(tf.feature_column.numeric_column(key='OriginRainfallInches'))
my_columns.append(tf.feature_column.numeric_column(key='OriginVisibility'))
my_columns.append(tf.feature_column.numeric_column(key='ArrivalWindSpeed'))
my_columns.append(tf.feature_column.numeric_column(key='ArrivalRainfallInches'))
my_columns.append(tf.feature_column.numeric_column(key='isHoliday'))
my_columns.append(tf.feature_column.numeric_column(key='SCHEDULED_ARRIVAL_HOUR'))
my_columns.append(tf.feature_column.numeric_column(key='SCHEDULED_DEPARTURE_HOUR'))


# Define train/test split
train_size = int(len(df)*0.8)
print(train_size)

data = df[df.columns[:-1]]
labels = df['DELAYED']

train_data = data[:train_size]
train_labels = labels[:train_size]
test_data = data[train_size:]
test_labels = labels[train_size:]

# Define model using canned TF estimator
estimator = tf.estimator.DNNClassifier(
    feature_columns=my_columns,
    hidden_units=[10,30,10],
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001),
    n_classes = 2)

# Define train and test input functions
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_data,
    y=train_labels, 
    batch_size=64, 
    shuffle=True, 
    num_epochs=10)
test_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=test_data,
    y=test_labels,
    batch_size=64,
    shuffle=True,
    num_epochs=3)

# Train model
estimator.train(input_fn=train_input_fn, steps=20000)
# Evaluate model
score = estimator.evaluate(input_fn = test_input_fn)
print(score)
#print('Test score: ', score[0])
#print('Test accuracy: ', score[1])



# Delayed Flight defined as more than 15 minutes late
# 1. Base keras model, 80% training data, no shuffle, 3 epics, batch_size 100, rainfall, wind speed and visibility variables
# Test accuracy: 0.8333
# Test score: 0.432
# 
# 2. Relu activation function, otherwise same as (1)
# Test accuracy: 0.833
# Test score: 0.434
# 
# 3. Base keras model, 80% training data, no shuffle, 3 epics, batch_size 100, no weather variables
# Test accuracy: 0.833
# Test score: 0.4367
# 
# 4. Base keras model, 60% training data, no shuffle, 3 epics, rainfall, wind speed, visibility variables
# Test accuracy: 0.818
# Test score: 0.455
# 
# 5. Same as (4) but shuffle = True
# Test accuracy: 0.812
# Test score: 0.462
# 
# Delayed Flight defined as more than 30 minutes late
# 1. Relu activation function, otherwise same as (1)
# Test accuracy: 0.900
# Test score: 0.31
# 
# 2. Relu activation function, retain all weather variables
# Test accuracy 0.900
# Test score: 1.605
# 
# 3. Sigmoid activation function, retain all weather variables
# Test accuracy: 0.900
# Test score: 0.313
