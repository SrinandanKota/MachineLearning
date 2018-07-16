Naive Bayes classifiers have been successfully applied to classifying text documents. In this lab assignment, you will implement the 
Naive Bayes algorithm to solve the \20 Newsgroups" classification problem.

1 Data Set
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different
newsgroups. It was originally collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews[1] paper, though he did not
explicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of
machine learning techniques, such as text classification and text clustering. The data is organized into 20 different newsgroups, each
corresponding to a different topic. Here is a list of the 20 newsgroups:
alt.atheism
comp.graphics
comp.os.ms-windows.misc
comp.sys.ibm.pc.hardware
comp.sys.mac.hardware
comp.windows.x
misc.forsale
rec.autos rec.motorcycles
rec.sport.baseball rec.sport.hockey
sci.crypt
sci.electronics
sci.med
sci.space
soc.religion.christian
talk.politics.guns
talk.politics.mideast
talk.politics.misc
talk.religion.misc

The original data set is available at http://qwone.com/~jason/20Newsgroups/. In this lab, you won't need to process the original data
set. Instead, a processed version of the data set is provided (see 20newsgroups.zip).

This processed version represents 18824 documents which have been divided to two subsets: training (11269 documents) and testing (7505
documents). You will find six files: map.csv, train label.csv, train data.csv, test label.csv, test data.csv, vocabulary.txt. The
vocabulary.txt contains all distinct words and other tokens in the 18824 documents. train data.csv and test data.csv are formatted
"docIdx, wordIdx, count", where docIdx is the document id, wordIdx represents the word id (in correspondence to vocabulary.txt) and
count is the frequency of the word in the document. train label.csv and test label.csv are simply a list of label id's indicating which
newsgroup each document belongs to. The map.csv maps from label id's to label names.

2 What You Will Do

Learn your Naive Bayes classifier from the training data (train label.csv, train data.csv), then evaluate its performance on the
testing data (test label.csv, test data.csv). Specifically, your program will accomplish the following two tasks. 

2.1 Learn Naive Bayes Model

You will implement the multinomial model ("a bag of words" model) and in the learning phase estimate the required probability terms
using the training data.

![alt text](screenshot/image.png)
