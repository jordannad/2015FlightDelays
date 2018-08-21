
# coding: utf-8

# ### Deep Neural Network Classifier of 2015 Flight Delay + Weather Data
# 
# Analysis performed with TensorFlow using a custom deep learning network implemented in keras.


import numpy as np 
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import utils

def column_names_with_prefix(column_names, prefix):
  """Helper function to inspect the dummy columns."""
  return [col for col in column_names if col.startswith(prefix)]


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

# Expand categorical feature columns to one hot encoding
dummy_fields = ['DAY_OF_WEEK', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'AIRLINE', 'SCHEDULED_DEPARTURE_HOUR',
 'SCHEDULED_ARRIVAL_HOUR']
for each in dummy_fields:
    dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
    df=pd.concat([df, dummies], axis=1)

df.head(10)
df = df.drop(dummy_fields, axis=1)

# Define training and test data split
train_size = int(len(df)*0.8)
print(train_size)

data = df[df.columns[:-1]]
labels = df['DELAYED']

train_data = data[:train_size]
test_data = data[train_size:]
labels_keras = utils.to_categorical(labels, 2)
train_labels = labels_keras[:train_size]
test_labels = labels_keras[train_size:]


# Set hyper-parameters for Keras model
batch_size = 100
epochs=3
train_data.shape

# Build model
model=Sequential()
model.add(Dense(256,input_shape=(168,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, ))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', metrics=['accuracy'])

# Train model
history = model.fit(train_data, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1
                   )

# Evaluate model 
score = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=1)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])
