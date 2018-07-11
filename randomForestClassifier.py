import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Read in pre-cleaned data. 
# Data has no missing values, otherwise, specify numeric cols as floats
df = pd.read_csv('data/FlightsAndWeatherData_top50Airports_NoMissingData.txt', sep='\t')
df.head()
# df['YEAR'] = df['YEAR'].astype(int)
# df['MONTH'] = df['MONTH'].astype(int)
# df['DAY'] = df['DAY'].astype(int)
# df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].astype(int)

# Remove unnecessary columns as desired
df = df.drop(['YEAR', 'DATE', 'SCHEDULED_ARRIVAL', 'SCHEDULED_DEPARTURE',
	'OriginDewPoint', 'OriginRelHumidity', 'OriginWindDirection',
	'OriginAltimeter', 'OriginWindGusts', 'ArrivalTemp','ArrivalDewPoint',
	'ArrivalRelHumidity', 'ArrivalWindDirection', 'ArrivalAltimeter', 
	'ArrivalVisibility', 'ORIGIN_CONGESTION', 'DESTINATION_CONGESTION'], axis = 1)
cols = df.columns.tolist()
ncols = len(df.columns)
df.isnull().sum()

# Define y (label) variable
threshold = 15
df['DELAYED'] = np.where(df['ARRIVAL_DELAY']>threshold, 1, 0)
df = df.drop(['ARRIVAL_DELAY'], axis= 1)

# Separate into two dataframes to plot distributions of numeric variables
delayedFlights = df[df['DELAYED']==1]
ontimeFlights = df[df['DELAYED']==0]
plotCols = ['SCHEDULED_DEPARTURE_HOUR', 'SCHEDULED_ARRIVAL_HOUR','DISTANCE', 'OriginRainfallInches']
fig = plt.figure(figsize=(10,10))
for i in range(2):
  for j in range(2):
    ax = plt.subplot2grid((2,2),(i,j))
    _ = ax.hist(ontimeFlights[plotCols[2*i+j]], alpha=0.5)
    _ = ax.hist(delayedFlights[plotCols[2*i+j]], alpha=0.5)
    ax.title.set_text(plotCols[2*i+j])

# One hot encoding of categorical variables
# sklearn's one hot encoder does not handle text values
dummy_fields = ['DAY_OF_WEEK', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'AIRLINE', 'SCHEDULED_DEPARTURE_HOUR',
 'SCHEDULED_ARRIVAL_HOUR']
for each in dummy_fields:
    dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
    df=pd.concat([df, dummies], axis=1)
df.head(10)
df = df.drop(dummy_fields, axis=1)

# Define train and test data sets
df = shuffle(df)
train_size = int(len(df)*0.7)
print(train_size)
data = df.loc[:, df.columns != 'DELAYED']
labels = df['DELAYED']

train_data = data[:train_size]
train_labels = labels[:train_size]
test_data = data[train_size:]
test_labels = labels[train_size:]

model = RandomForestClassifier(n_estimators = 100, verbose = 3)

# 5-fold cross validation of model
score_acc =( 
    cross_val_score(model,train_data,
                train_labels,
                cv=5,scoring='accuracy') )
print("Accuracy: %0.2f (+/- %0.2f)" 
      % (score_acc.mean(), score_acc.std() * 2))

score_f1 =( 
    cross_val_score(model, train_data, train_labels,
                cv=5,scoring='f1') )
print("F1 Score: %0.2f (+/- %0.2f)" 
      % (score_f1.mean(), score_f1.std() * 2))

# Fit the classifier to the training set
model.fit(train_data, train_labels)

# Predict probabilities of classes on the test set
predictions = model.predict(test_data)

# Assess random forest classifier performance on test set
# True Positive (TP)
TP = np.sum(np.logical_and(predictions == 1, test_labels == 1))
# True Negative (TN)
TN = np.sum(np.logical_and(predictions == 0, test_labels == 0))
# False Positive (FP)
FP = np.sum(np.logical_and(predictions == 1, test_labels == 0))
# False Negative (FN)
FN = np.sum(np.logical_and(predictions == 0, test_labels == 1))

print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
