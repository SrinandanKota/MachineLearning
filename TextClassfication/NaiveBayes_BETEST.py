# Bayesian Estimate on Test data

import csv
import math


## Each row in data file is saved as a list to corresponding class from label file
## Dict[class: [docid,wordid,count],[],[],[]]

## Calulating Components from train_label file is the same for all the files

train_data={}

train_label_file=open('train_label.csv','r')
                    
with open('train_data.csv','rb') as training_data_file:
    
    train_lines=csv.reader(training_data_file)
    for train_line in train_lines:
        line=train_label_file.readline().strip('\n')
        train_data.setdefault(line,[])
        row0=''.join(train_line[0].strip('\n'))
        row1=''.join(train_line[1].strip('\n'))
        row2=''.join(train_line[2].strip('\n'))
        row2=(float)(row2)
        row_line=[row0,row1,row2]
        train_data[line].append(row_line)
        
training_data_file.close()
    
train_label_file.close()

## Remove keys with no values

if '' in train_data.keys():
    del train_data['']

## Calculate Class priors

print "class priors"

train_label_file=open('train_label.csv','r')
train_lines=train_label_file.readlines()
train_label_file.close()
total_class_samples=len(train_lines)


class_priors={}

for class_id,total_class in train_data.items():
    class_priors.setdefault(class_id,0)
    class_priors[class_id]=(float)((float)(len(train_data[class_id]))/(float)(total_class_samples))
    print "P(Omega =",class_id,")=", class_priors[class_id]


print "classifying test data using bayesin estimates"

train={}

test_label_file=open('test_label.csv','r')
                    
with open('test_data.csv','rb') as test_data_file:
    
    train_lines=csv.reader(test_data_file)
    for test_line in train_lines:
        line=test_label_file.readline().strip('\n')
        train.setdefault(line,[])
        row0=''.join(test_line[0].strip('\n'))
        row1=''.join(test_line[1].strip('\n'))
        row2=''.join(test_line[2].strip('\n'))
        row2=(float)(row2)
        row_line=[row0,row1,row2]
        train[line].append(row_line)
        
test_data_file.close()
    
test_label_file.close()

## Remove keys with no values

if '' in train.keys():
    del train['']

total_n=0

for class_id,class_val in train.items():
    for val in class_val:
        total_n=total_n+val[2]

## Calculating nk

train_nk={}
train_nk_val={}
for class_id,class_val in train.items():
    train_nk.setdefault(class_id,{})
    for val in class_val:
        train_nk_val.setdefault(val[1],0)
        train_nk_val[val[1]]=train_nk_val[val[1]]+val[2]
    train_nk[class_id]=train_nk_val


voc_file=open('vocabulary.txt','r')
voc_lines=voc_file.readlines()
voc_size=len(voc_lines)

ml_est_val={}
ml_est={}
bay_est_val={}
bay_est={}

for class_id,class_val in train.items():
    ml_est.setdefault(class_id,{})
    bay_est.setdefault(class_id,{})
    for val in class_val:
        ml_est_val.setdefault(val[1],0)
        bay_est_val.setdefault(val[1],0)
        bay_est_val[val[1]]=bay_est_val[val[1]]+math.log(((float)(train_nk[class_id][val[1]])+1)/(float)(total_n+voc_size))
    ml_est[class_id]=ml_est_val
    bay_est[class_id]=bay_est_val

## Storing documents and the words in the document

doc_id_word={}
a=[]

for class_id,class_val in train.items():
    for val in class_val:
        doc_id_word.setdefault(val[0],[])
        doc_id_word[val[0]]=doc_id_word[val[0]]+list(val[1])

## Storing logarithmic probabilities of each word in a documennt belonging to a particular class 

doc_class_prob={}
doc_class_prob_val={}
for class_id,class_val in train.items():
    doc_class_prob.setdefault(class_id,{})
    for val in class_val:
        doc_class_prob_val.setdefault(val[0],0)
        doc_class_prob_val[val[0]]=doc_class_prob_val[val[0]]+bay_est[class_id][val[1]]
        doc_class_prob[class_id][val[0]]=doc_class_prob_val[val[0]]



for class_id,class_val in train.items():
    for val in class_val:
        doc_class_prob[class_id][val[0]]=doc_class_prob[class_id][val[0]]+math.log(class_priors[class_id])   
    
## Calculating argmax and classifying each documents to a particular class

train_set_class={}
overall_samples=0
correct_classified=0

for doc_id in doc_id_word.keys():
    train_set_class.setdefault(doc_id,'1')
    maxv=float('-Inf')
    for class_id in train.keys():
        if doc_id in doc_class_prob[class_id].keys() and (doc_class_prob[class_id][doc_id])>(maxv):
            train_set_class[doc_id]=class_id
            maxv=doc_class_prob[class_id][doc_id]
                
## Calculating overall accuracy

for class_id in train.keys():
    overall_samples=overall_samples+len(train[class_id])


## Calculating group accuracy

group_class={}

for class_id,class_val in train.items():
    group_class.setdefault(class_id,0)
    for val in class_val:
        if(class_id==train_set_class[val[0]]):
            correct_classified=correct_classified+1
            group_class[class_id]=group_class[class_id]+1


print "overall accuracy",(float)((float)(correct_classified)/(float)(overall_samples))*100,"%"

print "group accuracy"
for class_id,lis in train.items():
    print "group(",class_id,")",(float)((float)(group_class[class_id])/(float)(len(train[class_id])))*100,"%"


## Calculating confusion matrix
    
train_confusion_mat_val={}
train_confusion_mat={}

for class_id in train.keys():
    train_confusion_mat.setdefault(class_id,{})
    for doc_id in train_set_class.keys():
        class_conf=train_set_class[doc_id]
        train_confusion_mat_val.setdefault(class_conf,0)
        train_confusion_mat_val[class_conf]=train_confusion_mat_val[class_conf]+1
    train_confusion_mat[class_id]=train_confusion_mat_val

## Printing confusion matrix

print "connfusion matrix"

for class_id in train.keys():
    print train_confusion_mat[class_id].values() 

