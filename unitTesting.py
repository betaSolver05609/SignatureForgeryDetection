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

np.random.seed(2017)

PATH="C:\\Users\\Dell inspiron\\Desktop\\FinalProject";
os.chdir(PATH+"\\ready")

genuine='sample2*.png'
forged='forged2*.png'

genuine=glob.glob(genuine)
forged=glob.glob(forged)

def op_cnn(results):
    for val in results:
        val=val.ravel()
        plt.plot(val)
    plt.legend(data)
    plt.show()


def W_init(shape, name=None):
    values=rng.normal(loc=0, scale=1e-2,size=shape)
    return K.variable(values, name=name)
def b_init(shape, name=None):
    values=rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values,name=name)


input_shape=(105,105,1)

convnet=Sequential()
convnet.add(Conv2D(64,(10,10), activation='relu', input_shape=input_shape, kernel_initializer=W_init, kernel_regularizer=l2(2e-4)))
convnet.add(LeakyReLU(alpha=0.1))
convnet.add(MaxPooling2D())
convnet.add(Dropout(0.25))
convnet.add(Conv2D(128,(7,7), activation='relu', kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init))
convnet.add(LeakyReLU(alpha=0.1))
convnet.add(MaxPooling2D())
convnet.add(Dropout(0.25))
convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(LeakyReLU(alpha=0.1))
convnet.add(MaxPooling2D())
convnet.add(Dropout(0.4))
convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(LeakyReLU(alpha=0.1))
convnet.add(MaxPooling2D())
convnet.add(Dropout(0.3))
convnet.add(Flatten())
convnet.add(Dense(4096, activation='relu', kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init))
convnet.add(Dense(10, activation='relu',kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init))

convnet.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


data=genuine+forged
batch_size=1
epochs=20
lab=list()
for f in data:
    if 'sample' in f:
        lab.append(1)
    elif 'forged' in f:
        lab.append(0)
    

results=list();
for filename in data:
    img=misc.imread(filename)
    train_X=img
    valid_X=img
    train_X=train_X.flatten()
    valid_X=valid_X.flatten()
    train_X=train_X.astype('float32')
    train_X=train_X.reshape(-1,105,105,1)/255
    valid_X=valid_X.astype('float32')
    valid_X=valid_X.reshape(-1,105,105,1)/255
    train_label=np.array(1).reshape(1,)
    valid_label=np.array(1).reshape(1,)
    train_label=to_categorical(train_label,10)
    valid_label=to_categorical(valid_label,10)
    convnet_train = convnet.fit([valid_X], train_label, batch_size=batch_size,epochs=epochs,verbose=1)
    results.append(convnet.predict(valid_X, verbose=1))

op_cnn(results)

"""c=results
c=c.reshape(6,10)
b=np.array([0,0,0,0,1,1])
b=b.reshape(6,1)
c=np.append(c,b,axis=1)

df=pd.DataFrame(c,columns=['1','2','3','4','5','6','7','8','9','10','y'])

X_train=df.ix[:,0:10]
y_train=np.ravel(df.y)

scale=StandardScaler().fit(X_train)
X_train=scale.transform(X_train)

model=Sequential()

model.add(Dense(12, activation='relu', input_shape=(10,)))
 
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)
"""
