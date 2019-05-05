#encoding=utf-8

import sys
import os
import numpy as np
import pandas as pd
import h5py
import random

print "read files"
df_train = pd.read_csv("../data/train.csv")

q1 = df_train["q1"].values
q2 = df_train["q2"].values
for i in range(0,q1.shape[0]):
    if q1[i]>q2[i]:
        q1[i],q2[i] = q2[i],q1[i]
df_train["q1"] = q1
df_train["q2"] = q2
print df_train.head()
print df_train.describe()

q1 = df_train["q1"].values
q2 = df_train["q2"].values
label = df_train["label"].values

rows = q1.shape[0]

dict_1 = dict()

for i in range(0,rows):
    if label[i] == 1:
        if dict_1.get(q1[i],-1) == -1:
            dict_1[q1[i]] = [q2[i]]
        else:
            dict_1[q1[i]].append(q2[i])

        if dict_1.get(q2[i],-1) == -1:
            dict_1[q2[i]] = [q1[i]]
        else:
            dict_1[q2[i]].append(q1[i])

    if i%5000 == 0:
        sys.stdout.flush()
        sys.stdout.write("#")
print 
print len(dict_1)

listxy = []
for x in dict_1:
    listx = dict_1[x]
    if len(listx) > 1:
        listy = listx[:]
        random.shuffle(listy)
        for x,y in zip(listx,listy):
            if x<y:
                listxy.append([1,x,y])
        random.shuffle(listy)
        for x,y in zip(listx,listy):
            if x<y:
                listxy.append([1,x,y])

    if i%5000 == 0:
        sys.stdout.flush()
        sys.stdout.write("#")

print
print len(listxy)

random.shuffle(listxy)

df1 = pd.DataFrame(listxy)
df1.columns = ["label","q1","q2"]

df1.to_csv("ext_train.csv",index=False)
print('Complete')

