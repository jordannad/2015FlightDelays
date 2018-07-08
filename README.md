# 2015FlightDelays
Data exploration in R and predicting flight delays with TensorFlow

Results from DNN Classifier with TensorFlow.

Delayed Flight defined as more than 15 minutes late
1. Base keras model, 80% training data, no shuffle, 3 epics, batch_size 100, rainfall, wind speed and visibility variables.
Test accuracy: 0.8333
Test score: 0.432
 
2. Relu activation function, otherwise same as (1).
Test accuracy: 0.833
Test score: 0.434

3. Base keras model, 80% training data, no shuffle, 3 epics, batch_size 100, no weather variables.
Test accuracy: 0.833
Test score: 0.4367
 
4. Base keras model, 60% training data, no shuffle, 3 epics, rainfall, wind speed, visibility variables.
Test accuracy: 0.818
Test score: 0.455

5. Same as (4) but shuffle = True.
Test accuracy: 0.812
Test score: 0.462
 
Delayed Flight defined as more than 30 minutes late.
1. Relu activation function, otherwise same as (1).
Test accuracy: 0.900
Test score: 0.31

2. Relu activation function, retain all weather variables.
Test accuracy 0.900
Test score: 1.605
 
3. Sigmoid activation function, retain all weather variables.
Test accuracy: 0.900
Test score: 0.313
