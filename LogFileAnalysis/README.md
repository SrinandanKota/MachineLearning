# MachineLearning - Distributed Log File Analysis

Introduction 

Distributed systems debugging pose unique challenges for software professionals. It requires years of expertise and analysis to
point out an error in a distributed environment. One of the most common approaches in doing this is to gain insight into the system
activity of such a system and to analyze system logs. This can be quite tedious and in recent years machine learning approaches have
been used to analyze such log files to assist developers to identify issues in such systems and develop faster solutions. These log
files are usually stored in a text format and this makes applying Data Mining techniques very suitable for analysis of such files. We
can apply machine learning techniques to classify the events and try to find errors or detect an anomaly in the system that will help
the engineer to identify the issues. This will in turn lead to faster generalizations. These results can be used to repair the system.
We will discuss the latest technique of using Recurrent Neural Networks used for such analysis and develop a proof of concept Long Short
Term Memory Recurrent Neural Network model and observe its performance by feeding it a sample HDFS System log file to predict the
sequence of events in the file to help identify errors in the system.

Description

The code uses data preparation techniques on a sample system log file of a Hadoop Distributed File System and the implements a Long
Short Term Memory Recurrent Neural Network to predict the next event in the log file given previous input sequences and measure the
performance of this model for this prediction task. 

The events in a log file can be treated as a form of sequential data, we can use LSTM RNNs for predicting the next event given previous
events. LSTM RNNs have good retention capabilities and can deliver very good results for sequential data.  

Fetures of Log files and data preperation

A log file contains data that is produced automatically by the system and stores the information about the events that are taking place
inside the operating system. These files have data that are stored at different time intervals about the events that occur in the system
and are written to a file in a particular sequence. Many software applications and systems produce log files. There are many log files
like transaction log file, event log file, audit log file, server logs, etc.

A System log file which usually contains information like where, when and why, i.e.,IP-Address, Timestamp and the log message. There are
three main steps in doing log analysis- collection of data, structuring of data and analysis of data [8] which we will use to prepare
the data set. 

a sample of the System log file of a Hadoop Distributed File System. Some of the features found in a log file of a HDFS system are
timestamp, priority of the logging event, category of the logging event, the log message and platform dependent line characters.
To treat the events in the log file as a sequence we drop the extra information of the log messages and extract a part of the message
which can identify the message and treat this extracted information as a sequence of messages or events. These extracted messages have
the same order as the one in which they were recorded in the system log. 

For the purpose of this paper we have used a sample log file with recorded log messages of a Hadoop system and this file was among the
others that were used in [1] for detecting system problems. A sample log file having the messages of the Hadoop System is shown below in
the Figure 1.

Figure_1

Each event has a few attributes in the message associated with it and to treat this as a sequence prediction problem we extract a part
of the message which uniquely identifies it and treat it as a sequence of events. We can ignore the timesteps associated with each
message and just consider the messages as sequential data to be input to the LSTM recurrent neural networks. Following this technique,
we extract a part of the message that is associated with each message which uniquely identifies the event, like the number appearing
after the timestamp in the same figure. After applying these techniques to the system log file, we obtain sequential data and a visual
representation of this one-to-one sequence is shown in Fig 2 for the first few events in the log file. The representation indicates
that the first few events in the file appear in the order “978”, ”963”, “963”,”228”,”353” and “509”. The message can then be identified
from the original file by using the predicted sequence of the LSTM model.

Figure_2

PROOF OF CONCEPT MODEL

A.	Initial Processing of input data
After pre-processing the data, we will have a sequence of events that is made available from the log file. This will be our dataset to
the LSTM recurrent neural network model. Neural Networks model numbers, so we need to map the sequence of events to integer values. This
is implemented using the Keras  in Python. We will also create a reverse mapping for converting the predictions to our original sequence
of events. As we are developing a one-to-one model to predict the sequences of events, we will be using an input length of 1 and the
steps for processing such sequence predictions are discussed in [2]. At the beginning of the processed log file we have the sequence of
“978” and the true event after this is “963” . The model will try to predict the sequence “963” when the input to it is “978”. The input
data will be in the form of a NumPy  array which is to be converted into a format expected by LSTM networks, that is [sample, time steps,
features]. Once reshaped into this format we can then normalize the input integers to the range from 0-to-1, the range of
sigmoid activation functions used by LSTM network. This can then be treated as a sequence classification task, where each of the events
in the processed file represents a different class. The output can be converted to a one hot encoding , using the Keras
built-in functions. These steps will allow us to fit different LSTM models.


B.	LSTM ONE-SEQUENCE TO ONE-SEQEUNCE MAP

The code creates input and output pairs, on which the networks are trained and this is done by defining an input sequence length of
one and then reading input event sequences from the input file after pre-processing. The network will learn the mapping of events, but
this context will not be available to the network when making predictions.

Conclusion

A simple model of LSTM recurrent neural network is implemented to predict the sequence of events in a sample log file of a Hadoop System
in order to help the engineers to prepare for an event if it is an error. LSTM RNNs are able to store information for a good amount of
time, are resistant to noise and their system parameters are trainable, making it a very popular choice for classification tasks as
discussed in []. We have tried to implement a model that can predict the next event in a log file based on the previous events which
are not possible to obtain with great accuracy using some of the machine learning techniques. This is a very recent development in the
field of log analytics and this paper implements a simple model to demonstrate its working and it is  found that it works reasonably
well for small amounts of data and without much tuning of the parameters of the LSTM model




[1] Wei Xu, Ling Huang, Armando Fox, David Patterson, and Michael I. Jordan,” Detecting Large-Scale System Problems by Mining Console
Logs”.
[2] Understanding Stateful LSTM Recurrent Neural Networks in Python with Keras (Online available at:
https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/ )
[]



