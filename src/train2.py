#encoding=utf-8

import sys
import os
import numpy as np
import pandas as pd
import h5py
import random

# python train1.py -train valid=5
if len(sys.argv) <= 1:
    print "-train or -predict"
    sys.exit(0)
if sys.argv[1] != "-train" and sys.argv[1] != "-predict":
    print "-train or -predict"
    sys.exit(0)
print sys.argv[1][1:]

valid_fold = 0
if len(sys.argv)==3:
    valid_fold = int(sys.argv[2][6])
print "valid fold: " + str(valid_fold)

print "read files"
df_question = pd.read_csv("../data/question.csv")
# words 39, chars 58

df_train = pd.read_csv("../data/train.csv")
df_train = df_train.sample(frac=1,random_state=2018)

df_ext_train = pd.read_csv("./ext_train.csv")
temp_q1 = df_train["q1"].values.copy()
temp_q2 = df_train["q2"].values.copy()
np.random.shuffle(temp_q1)
np.random.shuffle(temp_q2)
temp_df = pd.DataFrame()
temp_df["label"] = np.zeros(temp_q1.shape[0],dtype=int)
temp_df["q1"] = temp_q1
temp_df["q2"] = temp_q2
temp_df = temp_df.sample(n=int(df_ext_train.shape[0]*0.8))
df_train = pd.concat([df_train,df_ext_train,temp_df])
df_train = df_train.drop_duplicates(["q1","q2"]).reset_index(drop=True)

df_test = pd.read_csv("../data/test.csv")

print "merge"
df_train = pd.merge(df_train,df_question,left_on="q1",right_on="qid",how="left")
df_train.rename(columns={'words':'words1'}, inplace = True)
df_train = pd.merge(df_train,df_question,left_on="q2",right_on="qid",how="left")
df_train.rename(columns={'words':'words2'}, inplace = True)
df_train = df_train[["label","words1","words2"]]

df_test = pd.merge(df_test,df_question,left_on="q1",right_on="qid",how="left")
df_test.rename(columns={'words':'words1'}, inplace = True)
df_test = pd.merge(df_test,df_question,left_on="q2",right_on="qid",how="left")
df_test.rename(columns={'words':'words2'}, inplace = True)
df_test = df_test[["words1","words2"]]

#----------------------------------------------------
print "build train token"
padding = 20891
rows = df_train.shape[0]
DOC_WORDS = 40
token_w1 = np.zeros([rows,DOC_WORDS],dtype=np.int32)
token_w2 = np.zeros([rows,DOC_WORDS],dtype=np.int32)
token_w1[:,:] = padding
token_w2[:,:] = padding

for i in range(0,rows):
    s = df_train["words1"][i]
    s = s.split(" ")
    c = 0
    for x in s:
        t = x.strip()
        t = int(t[1:])
        token_w1[i,c] = t
        c = c + 1
        if c == DOC_WORDS:
            break

for i in range(0,rows):
    s = df_train["words2"][i]
    s = s.split(" ")
    c = 0
    for x in s:
        t = x.strip()
        t = int(t[1:])
        token_w2[i,c] = t
        c = c + 1
        if c == DOC_WORDS:
            break

label = df_train["label"].values.astype(np.float32)

#----------------------------------------------------
print "build test token"
#padding = 20891
rows = df_test.shape[0]
#DOC_WORDS = 40
test_token_w1 = np.zeros([rows,DOC_WORDS],dtype=np.int32)
test_token_w2 = np.zeros([rows,DOC_WORDS],dtype=np.int32)
test_token_w1[:,:] = padding
test_token_w2[:,:] = padding

for i in range(0,rows):
    s = df_test["words1"][i]
    s = s.split(" ")
    c = 0
    for x in s:
        t = x.strip()
        t = int(t[1:])
        test_token_w1[i,c] = t
        c = c + 1
        if c == DOC_WORDS:
            break

for i in range(0,rows):
    s = df_test["words2"][i]
    s = s.split(" ")
    c = 0
    for x in s:
        t = x.strip()
        t = int(t[1:])
        test_token_w2[i,c] = t
        c = c + 1
        if c == DOC_WORDS:
            break

#----------------------------------------------------
print "build valid data"

rows = token_w1.shape[0]

indices = np.arange(rows)
indices = [x for x in indices if x % 10 != valid_fold]

train_x1 = token_w1[indices]
train_x2 = token_w2[indices]
train_y = label[indices]

indices = np.arange(rows)
indices = [x for x in indices if x % 10 == valid_fold]

valid_x1 = token_w1[indices]
valid_x2 = token_w2[indices]
valid_y = label[indices]

#----------------------------------------------------
test_x1 = test_token_w1
test_x2 = test_token_w2

#----------------------------------------------------
print "read embedding"
f = open('../data/word_embed.txt', 'r')
i = 0
max_features = padding + 1 
embedding_dims = 300

weights = np.zeros([max_features,embedding_dims],dtype=np.float32)
for line in f:
    s = line.split(' ')[1:]
    for j in range(300):
        weights[i,j] = float(s[j])

    if i%5000 == 0:
        sys.stdout.flush()
        sys.stdout.write("#")
    i=i+1
print
print "read embedding OK"

#----------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

from keras.models import Model
from keras.models import Sequential

from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint,Callback
from keras.callbacks import LearningRateScheduler

from keras.optimizers import Adam
from keras.engine import Layer, InputSpec

from keras import backend as K

import tensorflow as tf

#----------------------------------------------------
# define model
maxlen = DOC_WORDS
embedding_dims = 300

input1 = Input(shape=(maxlen,), dtype='int32')
input2 = Input(shape=(maxlen,), dtype='int32')

# shared encoder
text_input = Input(shape=(maxlen,), dtype='int32')
embedding_layer = Embedding(max_features,
                    embedding_dims,
                    trainable=True)
x = embedding_layer(text_input)
x = TimeDistributed(Dense(150, activation='relu'))(x)

xlstm = CuDNNLSTM(150, return_sequences=True)(x)
xlstm1 = GlobalMaxPooling1D()(xlstm)
xa = concatenate([xlstm, x])

xconv1 = Convolution1D(filters=100,
                    kernel_size=1,
                    padding='same',
                    activation='relu')(xa)
xconv1 = GlobalMaxPooling1D()(xconv1)

xconv2 = Convolution1D(filters=100,
                    kernel_size=2,
                    padding='same',
                    activation='relu')(xa)
xconv2 = GlobalMaxPooling1D()(xconv2)

xconv3 = Convolution1D(filters=100,
                    kernel_size=3,
                    padding='same',
                    activation='relu')(xa)
xconv3 = GlobalMaxPooling1D()(xconv3)

xconv4 = Convolution1D(filters=100,
                    kernel_size=4,dilation_rate=2,
                    padding='same',
                    activation='relu')(xa)
xconv4 = GlobalMaxPooling1D()(xconv4)

xconv5 = Convolution1D(filters=100,
                    kernel_size=5,dilation_rate=2,
                    padding='same',
                    activation='relu')(xa)
xconv5 = GlobalMaxPooling1D()(xconv5)

xconv6 = Convolution1D(filters=100,
                    kernel_size=6,
                    padding='same',
                    activation='relu')(xa)
xconv6 = GlobalMaxPooling1D()(xconv6)

xgru = CuDNNGRU(300, return_sequences=True)(xa)

x = concatenate([xconv1,xconv2,xconv3,xconv4,xconv5,xconv6,xlstm1])
x = Dropout(0.5)(x)
x = Dense(100)(x)
text_out = PReLU()(x)

text_encoder = Model(text_input, [text_out,xlstm,xgru])

# interaction
x1,l,lc = text_encoder(input1)
x2,r,rc = text_encoder(input2)

cross1 = Dot(axes=[2, 2], normalize=True)([l,r])
cross1 = Reshape((-1, ))(cross1)
cross1 = Dropout(0.5)(cross1)
cross1 = Dense(200)(cross1)
cross1 = PReLU()(cross1)

cross2 = Dot(axes=[2, 2], normalize=True)([lc,rc])
cross2 = Reshape((-1, ))(cross2)
cross2 = Dropout(0.5)(cross2)
cross2 = Dense(200)(cross2)
cross2 = PReLU()(cross2)

diff = subtract([x1,x2])
mul = multiply([x1,x2])
x = concatenate([x1,x2,diff,mul,cross1,cross2])

x = BatchNormalization()(x)

x = Dense(100)(x)
x = PReLU()(x)
x = Dropout(0.3)(x)

x = Dense(50)(x)
x = PReLU()(x)
x = Dropout(0.2)(x)

out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[input1,input2], outputs=out)

model.compile(loss='binary_crossentropy', optimizer='adam')

#----------------------------------------------------------------
# checkpoint

class my_eval(Callback):
    def on_train_begin(self, logs={}):
        self.best_score = 1

    def on_epoch_end(self, batch, logs={}):
        print
        print "eval begin"

        x = logs["val_loss"]
        print logs,x

        if x<self.best_score :
            self.best_score = x
            if x<0.8:
                self.model.save_weights("bestmodel.h5")
                print "save best weights."

        print
        print

eval_valid = my_eval()

def my_schedule(t):
    s = np.array([0.0008,0.0008,0.0007,0.0007,0.0006,
                  0.0006,0.0005,0.0004,0.0003,0.0002,
                  0.0001,0.0003,0.0002,0.0001,0.0003,
                  0.0002,0.0001,0.0003,0.0002,0.0001])
    print "epoch/Lr: ",t+1,s[t]
    return s[t]

callbacks_list = [eval_valid , LearningRateScheduler(schedule=my_schedule)]

#----------------------------------------------------------------
# train model
embedding_layer.set_weights([weights])

if os.path.exists('bestmodel.h5'):
    model.load_weights('bestmodel.h5')
    print "load weight ok"

if sys.argv[1] == "-train":
    model.fit([train_x1,train_x2], train_y,
          epochs=10, batch_size=88,validation_data=([valid_x1,valid_x2],valid_y),
          verbose=1,callbacks=callbacks_list)

if sys.argv[1] == "-predict":
    y_pred1 = model.predict([test_x1,test_x2],verbose=1,batch_size=640)
    y_pred1 = y_pred1[:,0]

    y_pred2 = model.predict([test_x2,test_x1],verbose=1,batch_size=640)
    y_pred2 = y_pred2[:,0]

    eps = 1e-7
    y_pred = (y_pred1.clip(eps,1-eps)+y_pred2.clip(eps,1-eps))/2.0

    df1= pd.DataFrame()
    df1["y_pre"] = y_pred
    print df1.head()
    print df1.describe()

    df1.to_csv("submission.csv",index=False,float_format = '%.9f')

print('Complete')
