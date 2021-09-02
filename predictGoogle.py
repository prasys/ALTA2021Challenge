import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import logging
import pickle5 as pickle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import sklearn
import argparse
import os,sys
import wandb
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
import numpy as np
import time
import calendar


def parseDocument(filename,documentID,number):
    file1 = open(filename, 'r')
    for line in file1:
        temp = line.split()
        if (len(temp)-2) == 1: #if we only have 1 document collection
            documentID.append(temp[number])
        elif (len(temp)-2) > 1:
                for count in range(2,len(temp)):
                    documentID.append(temp[number])
    file1.close()


GradeBandC = False
multiLabel = True

if multiLabel == False:
    if GradeBandC:
        grade0 = 'B'
        grade1 = 'C'
        replaceValue1 = 0
        replaceValue2 = 1
    else:
        grade0 = 'A'
        grade1 = 'B'
        replaceValue1 = 0
        replaceValue2 = 1

    


eval_df = pd.read_pickle('devMinA.h5') #devMinAS.h5 #devMinB
if multiLabel == False:
    eval_df['tagged'].replace({'A': 0, 'B': 1, 'C' :2}, inplace=True)
docID = []
parseDocument("devtestset.txt",docID,2)
eval_df['DocID'] = docID
docID = []
parseDocument("devtestset.txt",docID,0)
eval_df['CollectionID'] = docID
eval_df.to_csv('output_to_google.csv')
eval_df['count'] = eval_df['CollectionID']  # copy column for counting

eval_df = eval_df.groupby(['CollectionID']).agg({
    'tagged' : np.sum,
    'count': np.size
}).reset_index()

eval_df['tagged'] /= eval_df['count']
eval_df = eval_df.drop(['count'], axis=1)
if multiLabel == False:
    eval_df['tagged'] = eval_df['tagged'].mask(eval_df['tagged'] < 0.49, 0) #make the probability of the class to be A/B
    eval_df['tagged'] = eval_df['tagged'].mask(eval_df['tagged'] > 0.89, 2) # make the probability of the class to be C/B
    eval_df['tagged'] = eval_df['tagged'].mask(eval_df['tagged'] > 0.49, 1) #make the probability of the class to be A/B
    eval_df['tagged'].replace({1: 'B', 0: 'A', 2: 'C'}, inplace=True)
else:
    eval_df['predicted'] = eval_df['predicted'].mask(eval_df['predicted'] < 0.49, 0) #make the probability of the class to be A/B
    eval_df['predicted'] = eval_df['predicted'].mask(eval_df['predicted'] > 0.49, 1) #make the probability of the class to be A/B
    eval_df['predicted'].replace({1: grade1, 0: grade0}, inplace=True)

eval_df.to_csv('answer.txt',header=None,index=False,sep="\t")
ts = calendar.timegm(time.gmtime())
exportFileName = "export" + str(ts) +".zip"
zipObj = ZipFile(exportFileName, 'w')
zipObj.write('answer.txt')
zipObj.close()
