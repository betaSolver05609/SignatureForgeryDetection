# -*- coding: utf-8 -*-
"""
Created on Sun May 06 01:39:11 2018

@author: Someindra Kumar Singh
"""
import os
import glob
import numpy as np
from scipy import misc
from PreprocessingEngine import PreprocessingEngine
from Classifier import ClassifierDesign
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#Setting the root path
PATH="C:\\Users\\Dell inspiron\\Desktop\\FinalProject"
test_result=[np.array([[    0.     ,     0.     , 23720.057  ,     0.     ,   146.66452,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],
      dtype='float32'), np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype='float32'), np.array([[    0.   ,     0.   , 16616.203,     0.   ,     0.   ,     0.   ,
            0.   ,     0.   ,     0.   ,     0.   ]], dtype='float32'), np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype='float32'), np.array([[    0.    ,     0.    , 23272.121 ,     0.    ,     0.    ,
            0.    ,   327.3439,     0.    ,     0.    ,     0.    ]],
      dtype='float32'), np.array([[122.80974 ,   0.      ,   0.      ,   0.      ,   0.      ,
          0.      , 172.51654 ,   0.      ,   0.      ,  46.130882]],
      dtype='float32')]
test_result=np.array(test_result)
#Getting forged data from a dataset
def forged_data(folder,ext):
    re="forged2"+"*"+ext
    os.chdir(PATH+folder)
    return glob.glob(re)
#getting genuine data from a dataset
def genuine_data(folder,ext):
    re="sample2"+"*"+ext
    os.chdir(PATH+folder)
    return glob.glob(re)
#Preprocessing the data returning a numpy array
def preprocess(filename):
    os.chdir(PATH+"\\dataset")
    img=misc.imread(filename)
    p=PreprocessingEngine(img)
    img=p.getGrayImage();
    img=p.ScaleImage();
    img=p.reduceNoise();
    img=p.eliminateBackground();
    return img
#Saving the file in the desired location
def save_ready(FILENAME, img):
    os.chdir(PATH+'\\dummyReady')
    img=img.astype(np.float32)
    misc.imsave(FILENAME.replace('.jpg','.png'), img)
#returning a numpy array of the said image
def extractImage(filename,folder):
    os.chdir(PATH+folder)
    img=misc.imread(filename)
    return img
#Creating the test set
def prepare_test_set(data,label):
    c=label
    dim=int(c.shape[0])
    c=c.reshape(dim,10)
    #keeping reference same
    p=data
    dim1=int(c.shape[0])
    p=p.reshape(dim1,10)
    ref=list()
    for i in range(dim1):
        ref.append(p[1])
    ref=np.array(ref)
    ref=ref.reshape(10,dim)
    X_train=np.concatenate((c,ref.T), axis=1)
    os.chdir(PATH+'\\dummy_test_set')
    forged_data=glob.glob("forged"+"*"+".png")
    genuine_data=glob.glob("sample"+"*"+".png")
    complete_dataset=forged_data+genuine_data
    label=list()
    for filename in complete_dataset:
        if 'forged' in filename:
            label.append(0)
        else:
            label.append(1)
    b=np.array(label)
    b=b.reshape(dim,1)
    c=np.append(X_train,b,axis=1)
    p=list()
    for i in range(20):
        p.append(str(i))
    p.append('y')
    df=pd.DataFrame(c,columns=p)
    X_train=df.ix[:,0:20]
    scale=StandardScaler().fit(X_train)
    X_test=scale.transform(X_train)
    return X_test


#Standardizing the data for training
def prepare_data(data):
    c=data
    dim=int(c.shape[0])
    c=c.reshape(dim,10)
    ref=list()
    for i in range(dim):
        ref.append(c[1])
    ref=np.array(ref)
    ref=ref.reshape(10,dim)
    X_train=np.concatenate((c,ref.T), axis=1)
    os.chdir(PATH+'\\dummyReady')
    forged_data=glob.glob("forged"+"*"+".png")
    genuine_data=glob.glob("sample"+"*"+".png")
    complete_dataset=forged_data+genuine_data
    label=list()
    for filename in complete_dataset:
        if 'forged' in filename:
            label.append(0)
        else:
            label.append(1)
    b=np.array(label)
    b=b.reshape(dim,1)
    c=np.append(X_train,b,axis=1)
    p=list()
    for i in range(20):
        p.append(str(i))
    p.append('y')
    df=pd.DataFrame(c,columns=p)
    X_train=df.ix[:,0:20]
    y_train=np.ravel(df.y)
    scale=StandardScaler().fit(X_train)
    X_train=scale.transform(X_train)
    return X_train,y_train

folder='\\dataset'
ext='.jpg'
forged=forged_data(folder,ext)
genuine=genuine_data(folder,ext)
complete=forged+genuine
for filename in complete:
    a=preprocess(filename)
    save_ready(filename, a)
folder='\\dummyReady'
ext='.png'
forged=forged_data(folder,ext)
genuine=genuine_data(folder,ext)
complete=forged+genuine
label=list()
for filename in complete:
    m=re.search('\d', filename)
    label.append(filename[m.start()])
image_list=list()
for filename in complete:
    image_list.append(extractImage(filename,folder))
x=len(image_list)
result=list()
for i in range(x):
    img=image_list[i]
    conv=ClassifierDesign(img, label[i])
    conv.train()
    result.append(conv.main_conv())

deep_copy=result
result=np.array(result)
data=result
label=test_result
X_train, y_train=prepare_data(data)
X_test=prepare_test_set(data,label)

model=Sequential()
model.add(Dense(30, activation='relu', input_shape=(20,)))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
         optimizer='adam',
         metrics=['accuracy'])
model.fit(X_train, y_train,epochs=100, batch_size=1, verbose=1)

os.chdir(PATH+'\\dummy_test_set')
forged_data=glob.glob("forged"+"*"+".png")
genuine_data=glob.glob("sample"+"*"+".png")
complete_dataset=forged_data+genuine_data
a=model.predict(X_test)
b=np.ravel(a)
agg=0
print('Name\t\tstatus\t\tprediction')
for i in range(len(complete_dataset)):
    status='forged'
    if 'forged' in complete_dataset[i]:
        status='forged'
    else:
        status='genuine'
    pred='forged'
    if b[i]>=0.5:
        pred='genuine'
    if status!=pred:
        agg=agg+1
    print(str(complete_dataset[i])+'\t'+status+'\t\t'+pred)
acc=((6-agg)/6*100)
print("Accuracy="+str(acc)+"%")



    

    




