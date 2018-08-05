import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import numpy.random as rng
import numpy as np
from scipy import misc
from keras import backend as K
from keras.regularizers import l2
from keras.utils import to_categorical
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(2018)

class ClassifierDesign(object):
    def __init__(self, img, label):
        self.img=img
        self.label=label
        self.input_shape=(105,105,1)
        self.convnet=Sequential()
        self.convnet.add(Conv2D(64,(10,10), activation='relu', input_shape=self.input_shape, kernel_initializer=self.W_init, kernel_regularizer=l2(2e-4)))
        self.convnet.add(LeakyReLU(alpha=0.1))
        self.convnet.add(MaxPooling2D())
        self.convnet.add(Dropout(0.25))
        self.convnet.add(Conv2D(128,(7,7), activation='relu', kernel_regularizer=l2(2e-4), kernel_initializer=self.W_init, bias_initializer=self.b_init))
        self.convnet.add(LeakyReLU(alpha=0.1))
        self.convnet.add(MaxPooling2D())
        self.convnet.add(Dropout(0.25))
        self.convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=self.W_init,kernel_regularizer=l2(2e-4),bias_initializer=self.b_init))
        self.convnet.add(LeakyReLU(alpha=0.1))
        self.convnet.add(MaxPooling2D())
        self.convnet.add(Dropout(0.4))
        self.convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=self.W_init,kernel_regularizer=l2(2e-4),bias_initializer=self.b_init))
        self.convnet.add(LeakyReLU(alpha=0.1))
        self.convnet.add(MaxPooling2D())
        self.convnet.add(Dropout(0.3))
        self.convnet.add(Flatten())
        self.convnet.add(Dense(4096, activation='relu', kernel_regularizer=l2(2e-4), kernel_initializer=self.W_init, bias_initializer=self.b_init))
        self.convnet.add(Dense(10, activation='relu',kernel_regularizer=l2(2e-4), kernel_initializer=self.W_init, bias_initializer=self.b_init))
        self.convnet.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    def W_init(self,shape, name=None):
        values=rng.normal(loc=0, scale=1e-2,size=shape)
        return K.variable(values, name=name)
    def b_init(self,shape, name=None):
        values=rng.normal(loc=0.5, scale=1e-2, size=shape)
        return K.variable(values,name=name)
    def prepare_data(self):
        train_X=self.img
        valid_X=self.img
        train_X=train_X.flatten()
        valid_X=valid_X.flatten()
        train_X=train_X.astype('float32')
        train_X=train_X.reshape(-1,105,105,1)/255
        valid_X=valid_X.astype('float32')
        valid_X=valid_X.reshape(-1,105,105,1)/255
        train_label=np.array(self.label).reshape(1,)
        valid_label=np.array(self.label).reshape(1,)
        train_label=to_categorical(train_label,10)
        valid_label=to_categorical(valid_label,10)
        return train_X, train_label, valid_X, valid_label
    def train(self):
        epochs=20
        batch_size=1
        train_X, train_label, valid_X, valid_label=self.prepare_data()
        self.convnet.fit([valid_X], train_label, batch_size=batch_size,epochs=epochs,verbose=1)
    def main_conv(self):
        train_X, train_label, valid_X, valid_label=self.prepare_data()
        result=self.convnet.predict(valid_X, verbose=1)
        return result
        
        
        
    