# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:18:40 2018

@author: SRINANDAN KOTA
"""

###
import pandas as pd
from pandas import read_csv
from pandas import datetime
from numpy import array
import numpy
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import argmax
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# convert an array of values into a dataset matrix
#def create_dataset(dataset):
#    dataX, dataY = [], []
#    seq=list()
#    array_seq=numpy.array(len(dataset),dtype=object)
#    for sval in range(len(dataset)-1):
#        if sval==0:
#            seq.append(["000",dataset[sval+1]])
#        else:
#            seq.append([dataset[sval],dataset[sval+1]]) 
#    print(seq)
#    array_seq=array(seq)
#    X,y=array_seq[:,0],array_seq[:,1]
#    X=X.reshape((len(X),1,1))
#    return X, y
     
    
    
# fix random seed for reproducibility
numpy.random.seed(7)

# reading hdfs log file
hdfs_log_val=read_csv('hadoop-log.txt',delimiter=",", names=["ix1", "ix2"])

# lists to store events
hdfs_log=list()
hdfs_events=list()

# datd preperation    
for index,rows in hdfs_log_val.iterrows():
    rows_event=rows["ix2"].split(" ")
    hdfs_log.append([rows["ix1"],rows_event[0]])
    hdfs_events.append(rows_event[0])
    

# process string to number format for processing
msg_to_int= dict((c,i) for i, c in enumerate(hdfs_events) )
#print(msg_to_int)
int_to_msg=dict((i,c) for i, c in enumerate(hdfs_events) )

# list to store real events
y_r=list()

# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []

print("original sequence")

for i in range(0, len(hdfs_events) - seq_length, 1):
    seq_in = hdfs_events[i:i + seq_length]
    seq_out = hdfs_events[i + seq_length]
    dataX.append([msg_to_int[char] for char in seq_in])
    dataY.append(msg_to_int[seq_out])
    y_r.append(seq_out) 
    print(seq_in,'->',seq_out)
    

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))

# normalize
X = X / float(len(hdfs_events))

y = np_utils.to_categorical(dataY)

# create and fit the model
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(256))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=1, verbose=0)

# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
#print("Model Accuracy: %.2f%%" % (scores[1]*100))


print("predicted sequence")

# list to store predicted sequence by the model
y_p=list()

for pattern in dataX:
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(hdfs_events))       
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_msg[index]
    seq_in = [int_to_msg[value] for value in pattern]
    y_p.append(result)
    print(seq_in,'->',result)
            
c_m=confusion_matrix(y_r,y_p)
#print(c_m)

c_rep=classification_report(y_r,y_p)
#print(c_rep)