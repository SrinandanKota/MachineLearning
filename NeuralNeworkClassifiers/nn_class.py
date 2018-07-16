# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

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
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping

# Define baseline model


# ReLU,one layer,sixteen inputs, zero learning,zero momentum, no scaling model

def relu_model_1():
	# Create model
    model = Sequential()
    model.add(Dense(16, input_dim=64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ReLU,one layer,thirty two inputs, zero learning,zero momentum, no scaling model

def relu_model_2():
	# Create model
	model = Sequential()
	model.add(Dense(32, input_dim=64, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# ReLU,two layer,different inputs, zero learning,zero momentum, no scaling model

def relu_model_3():
	# Create model
    model = Sequential()
    model.add(Dense(32, input_dim=64, activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ReLU,one layer,sixteen inputs, some learning,zero momentum, no scaling model

def relu_model_4():
	# Create model
    sgd=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model = Sequential()
    model.add(Dense(16, input_dim=64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# ReLU,one layer,sixteen inputs, some learning,some momentum, no scaling model

def relu_model_5():
	# Create model
    sgd=optimizers.SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=False)
    model = Sequential()
    model.add(Dense(16, input_dim=64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# ReLU,mean squared error, one layer,sixteen inputs, zero learning,zero momentum, no scaling model

def relu_model_6():
	# Create model
    model = Sequential()
    model.add(Dense(16, input_dim=64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

# ReLU,mean squared error, one layer,thirty two inputs, zero learning,zero momentum, no scaling model

def relu_model_7():
	# Create model
	model = Sequential()
	model.add(Dense(32, input_dim=64, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

# ReLU,mean squared error, two layers,different inputs, zero learning,zero momentum, no scaling model

def relu_model_8():
	# create model
    model = Sequential()
    model.add(Dense(32, input_dim=64, activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

# ReLU,mean squared error, one layer,sixteen inputs, some learning,zero momentum, no scaling model

def relu_model_9():
	# Create model
    sgd=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model = Sequential()
    model.add(Dense(16, input_dim=64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model

# ReLU,mean squared error, one layer,sixteen inputs, some learning, some momentum, no scaling model

def relu_model_10():
	# create model
    sgd=optimizers.SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=False)
    model = Sequential()
    model.add(Dense(16, input_dim=64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model

# Find convergence time

def find_conv(model,x,y):
    start=time.time()
    model.fit(x,y,callbacks=[earlystop],validation_split=0.2)
    end=time.time()
    return end-start

# Find Scores

def find_scores(model,x,y):
    scores = model.score(x, y)
    return scores*100

# tanh function,one layer,sixteen inputs, zero learning,zero momentum, no scaling model

def tanh_fun_model_1():
	# Create model
    model = Sequential()
    model.add(Dense(16, input_dim=64, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# tanh function,one layer,thirty two inputs, zero learning,zero momentum, no scaling model

def tanh_fun_model_2():
	# Create model
	model = Sequential()
	model.add(Dense(32, input_dim=64, activation='tanh'))
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# tanh function,two layers,variable inputs, zero learning,zero momentum, no scaling model

def tanh_fun_model_3():
	# create model
    model = Sequential()
    model.add(Dense(32, input_dim=64, activation='tanh'))
    model.add(Dense(16,activation='tanh'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# tanh function,one layer,sixteen inputs, some learning,zero momentum, no scaling model

def tanh_fun_model_4():
	# Create model
    sgd=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model = Sequential()
    model.add(Dense(16, input_dim=64, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# tanh function,one layer,sixteen inputs, some learning,some momentum, no scaling model

def tanh_fun_model_5():
	# Create model
    sgd=optimizers.SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=False)
    model = Sequential()
    model.add(Dense(16, input_dim=64, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("optdigits.csv", delimiter=",")
dataset = dataframe.values

train_x = dataset[:,0:64]
train_y = dataset[:,64]


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_y)
encoded_y = encoder.transform(train_y)
# convert integers to dummy variables (i.e. one hot encoded)
train_ocy = np_utils.to_categorical(encoded_y)


# Load test data
dataframe_test= pandas.read_csv("test_optdigits.csv", delimiter=",")
dataset_t=dataframe_test.values

test_x= dataset_t[:,0:64]
test_y = dataset_t[:,64]

# encode class values as integers
encoder_t = LabelEncoder()
encoder_t.fit(test_y)
encoded_t_y = encoder.transform(test_y)
# convert integers to dummy variables (i.e. one hot encoded)
test_ocy = np_utils.to_categorical(encoded_t_y)


scale_train_x=scale(train_x)

scale_test_x=scale(test_x)

model=KerasClassifier(build_fn=relu_model_1, epochs=30, batch_size=5, verbose=0)

# define early stopping callback
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, \
                          verbose=0, mode='auto')

print("ReLu activation")
print("cross entropy")

print("Model with one layer")
print("Convergence speed is %f"%find_conv(model,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model,train_x,train_ocy))


print("Model after scaling input using standardization")
print("Convergence speed is %f"%find_conv(model,scale_train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model,scale_train_x,train_ocy))


model1=KerasClassifier(build_fn=relu_model_2, epochs=30, batch_size=5, verbose=0)


print("Model with one layer and more hidden inputs")
print("Convergence speed is %f"%find_conv(model1,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model1,train_x,train_ocy))


# This is the best model

model2=KerasClassifier(build_fn=relu_model_3, epochs=30, batch_size=5, verbose=0)

print("Model with two layers")
print("Convergence speed is %f"%find_conv(model2,scale_train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model2,scale_train_x,train_ocy))

# Printing Class report and Confusion Matrix for the best model here

# Train data

train_x_pred=model2.predict(scale_train_x)
cm_train=confusion_matrix(train_y,train_x_pred)
print("confusion matrix for training data with two layer model")
print(cm_train)
print("classfication report for training data with two layer model")
class_report_train=classification_report(train_y,train_x_pred)
print(class_report_train)

# Test data

test_x_pred=model2.predict(scale_test_x)
cm_test=confusion_matrix(test_y,test_x_pred)
print("confusion matrix for test data with two layer model")
print(cm_test)
print("classfication report for test data with two layer model")
class_report_test=classification_report(test_y,test_x_pred)
print(class_report_test)


model3=KerasClassifier(build_fn=relu_model_4, epochs=30, batch_size=5, verbose=0)


print("Model with leraning rate")
print("Convergence speed is %f"%find_conv(model3,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model3,train_x,train_ocy))


model4=KerasClassifier(build_fn=relu_model_5, epochs=30, batch_size=5, verbose=0)

print("Model with learning rate and momentum")
print("Convergence speed is %f"%find_conv(model4,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model4,train_x,train_ocy))


print("for tanh activation")
print("cross entropy")

model=KerasClassifier(build_fn=tanh_fun_model_1, epochs=30, batch_size=5, verbose=0)

print("Model with one layer")
print("Convergence speed is %f"%find_conv(model,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model,train_x,train_ocy))


print("Model after scaling input using standardization")
print("Convergence speed is %f"%find_conv(model,scale_train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model,scale_train_x,train_ocy))

model1=KerasClassifier(build_fn=tanh_fun_model_2, epochs=30, batch_size=5, verbose=0)


print("Model with one layer and more hidden inputs")
print("Convergence speed is %f"%find_conv(model1,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model1,train_x,train_ocy))


# This is the best model with tanh function

model2=KerasClassifier(build_fn=tanh_fun_model_3, epochs=30, batch_size=5, verbose=0)

print("Model with two layers")
print("Convergence speed is %f"%find_conv(model2,scale_train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model2,scale_train_x,train_ocy))

# Confusion matrix and class accuracy report

# Train data

train_x_pred=model2.predict(scale_train_x)
cm_train=confusion_matrix(train_y,train_x_pred)
print("confusion matrix for training data with two layer model")
print(cm_train)
print("classfication report for training data with two layer model")
class_report_train=classification_report(train_y,train_x_pred)
print(class_report_train)


# Test data

test_x_pred=model2.predict(scale_test_x)
cm_test=confusion_matrix(test_y,test_x_pred)
print("confusion matrix for test data with two layer model")
print(cm_test)
print("classfication report for test data with two layer model")
class_report_test=classification_report(test_y,test_x_pred)
print(class_report_test)



model3=KerasClassifier(build_fn=tanh_fun_model_4, epochs=30, batch_size=5, verbose=0)


print("Model with learning rate")
print("Convergence speed is %f"%find_conv(model3,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model3,train_x,train_ocy))


model4=KerasClassifier(build_fn=tanh_fun_model_5, epochs=30, batch_size=5, verbose=0)

print("Model with learning rate and momentum")
print("Convergence speed is %f"%find_conv(model4,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model4,train_x,train_ocy))

print("relu activation")

print("mean squared error")

model=KerasClassifier(build_fn=relu_model_6, epochs=30, batch_size=5, verbose=0)

print("Model with one layer")
print("Convergence speed is %f"%find_conv(model,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model,train_x,train_ocy))


print("Model after scaling input using standardization")
print("Convergence speed is %f"%find_conv(model,scale_train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model,scale_train_x,train_ocy))

model1=KerasClassifier(build_fn=relu_model_7, epochs=30, batch_size=5, verbose=0)


print("Model with one layer and more hidden inputs")
print("Convergence speed is %f"%find_conv(model1,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model1,train_x,train_ocy))


# This is the best model with relu function and mean square error function

model2=KerasClassifier(build_fn=relu_model_8, epochs=30, batch_size=5, verbose=0)

print("Model with two layers")
print("Convergence speed is %f"%find_conv(model2,scale_train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model2,scale_train_x,train_ocy))

# Confusion Matrix and class accuracy report

# Train data

train_x_pred=model2.predict(scale_train_x)
cm_train=confusion_matrix(train_y,train_x_pred)
print("confusion matrix for training data with two layer model")
print(cm_train)
print("classfication report for training data with two layer model")
class_report_train=classification_report(train_y,train_x_pred)
print(class_report_train)

# Test data

test_x_pred=model2.predict(scale_test_x)
cm_test=confusion_matrix(test_y,test_x_pred)
print("confusion matrix for test data with two layer model")
print(cm_test)
print("classfication report for test data with two layer model")
class_report_test=classification_report(test_y,test_x_pred)
print(class_report_test)

model3=KerasClassifier(build_fn=relu_model_9, epochs=30, batch_size=5, verbose=0)


print("Model with learning rate")
print("Convergence speed is %0.2f"%find_conv(model3,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model3,train_x,train_ocy))

model4=KerasClassifier(build_fn=relu_model_10, epochs=30, batch_size=5, verbose=0)

print("Model with learning rate and momentum")
print("Convergence speed is %0.2f"%find_conv(model4,train_x,train_ocy))
print("Accuracy is %0.4f"%find_scores(model4,train_x,train_ocy))
