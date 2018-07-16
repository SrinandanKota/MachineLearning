# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:50:08 2018

@author: SRINANDAN KOTA
"""


import numpy
import pandas
import time
import sys
from keras import optimizers
from keras import losses
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping


def model_cnn(x,y,dropout_1,dropout_2):
    model = Sequential()
    # Filter size can be changed here, it is 3x3 by default
    # Edit here
    # Best model
    model.add(Convolution2D(32, 3, 3 , input_shape=(8,8,1), activation='relu',border_mode = 'valid')) 
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th')) 
    model.add(Dropout(dropout_1))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_2))
    model.add(Dense(10, activation='softmax'))   
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5)
    start=time.time()
    model.fit(train_x,train_ocy, callbacks=[earlystop], batch_size=5, nb_epoch=25, 
                validation_split=0.2)
    end=time.time()
    print("Time to train the model is",end-start)
    return model

# Evaluate cnn model

def eval_cnn(model,x,y):
    scores = model.evaluate(x, y,verbose=0)
    return scores[1]*100

# Predict the output  

def model_pred(model,x,y):
    pred_x=model.predict_classes(x)
    pred_y=model.predict_classes(y)
    return pred_x,pred_y


# load dataset
    
dataframe = pandas.read_csv("optdigits.csv", delimiter=",")
dataset = dataframe.values

train_x = dataset[:,0:64]
train_y = dataset[:,64]


# Load test data

dataframe_test= pandas.read_csv("test_optdigits.csv", delimiter=",")
dataset_t=dataframe_test.values

test_x= dataset_t[:,0:64]
test_y = dataset_t[:,64]


# Reshaping data to fit the neural network model

train_x = train_x.reshape(-1,8,8,1)
test_x = test_x.reshape(-1,8,8,1)


# One-hot encoding of data

train_ocy=np_utils.to_categorical(train_y)
test_ocy=np_utils.to_categorical(test_y)


# Prepare the model

print("model1")
# Dropouts can be defined here, it is currently 0.3,0.6
# Edit here
# Best model
model1=model_cnn(train_x,train_ocy,0.3,0.6)
scores1=eval_cnn(model1,test_x,test_ocy)

print("score is %0.2f"%scores1)

pred_train_x, pred_test_x=model_pred(model1,train_x,test_x)


# Classfication reports and Confusion matrix for the best model 

# Train data

print("confusion matrix for training data")
conf_train=confusion_matrix(train_y,pred_train_x)
print(conf_train)

print("classification report")
class_train_report=classification_report(train_y,pred_train_x)
print(class_train_report)

# Test data

print("confusion matrix for testing data")
conf_test=confusion_matrix(test_y,pred_test_x)
print(conf_test)

print("classification report")
class_test_report=classification_report(test_y,pred_test_x)
print(class_test_report)

