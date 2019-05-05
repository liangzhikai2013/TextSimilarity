import numpy as np
import os
import glob
import math
import pickle
import datetime


import pandas as pd

files=[]
for filename in glob.glob(r'submission*.csv'):
    files.append(filename)

nb_files=len(files)


s=0
flag=1

for f in files:
    df_test=pd.read_csv(f)
    if flag==1:
        s=np.array(list(df_test['y_pre']))
        flag=0
    else:
        s=s+np.array(list(df_test['y_pre']))

df_test["y_pre"]=s/nb_files

df_test.to_csv("avg.csv",index=False,float_format='%0.9f')

print df_test.describe()

