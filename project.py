
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
def op_cnn(results,complete):
    for val in results:
        val=val.ravel()
        plt.plot(val)
    plt.legend(complete)
    plt.show()

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


op_cnn(result,complete)


    




