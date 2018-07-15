# MachineLearning - Neural Network Classfiers

In this assignment, you will experiment with the Neural Network classifier.

1 Dataset

This experiment will use the following data set from the UC Irvine Machine Learning Repository:
Optical Recognition of Handwritten Digits Data Set (use optdigits.names, optdigits.tra as training data, and optdigits.tes as test data)
(https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits).

2 Tasks

The experiment will use the Neural Network Classifier. It uses the softmax function for the output layer and uses 1-of-c output encoding
with target values such as (1, 0, 0, ...). The early stopping technique is used to decide when to stop training. For example, we can use
20% of training data in optdigits.tra as the validation set.  

1. Experiment with fully-connected feed-forward neural networks.

(a) Sum-of-squares error vs. cross-entropy error function. Use ReLU units for the hidden layers. For each of the two types of error
functions, different values of hyper-parameters are used, including number of hidden layers, number of hidden units in each layer, learning
rates, momentum rates, input scaling, and so on. We compare their classification accuracy and convergence speed (time till training stops).
A report of the experimental results, the best hyper-parameter values and a report for the best model learned, the corresponding
hyper-parameters and the performance including overall classification accuracy, class accuracy, and confusion matrix for both training and
testing data is recorded and the results ares discussed here.

(b) tanh vs. ReLU hidden units. Use the cross-entropy error function. For each of the two types of hidden units, the above experiments are
repeated, that is, experiment is conducted with different values of hyper-parameters and the results are reported. The results are discussed
here.

2. Experiment with convolutional networks (CNNs). 

This experiment uses the cross-entropy error function, and ReLU hidden units. The previous experiments are repeated, that is, experiment
with different values of hyper-parameters (note CNNs may have different types of hyper-parameters, eg. filter size) and the results are
reported. We then discuss the results.
