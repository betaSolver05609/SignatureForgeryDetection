# -*- coding: utf-8 -*-
"""
Created on Thu May 03 02:24:10 2018

@author: Dell inspiron
"""
import numpy as np
import pandas as pd;
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
import matplotlib.pyplot as plt
np.random.seed(2018)

history=History()
model=Sequential()


results=np.array([np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.  , 50029.36,     0.  ,     0.  ,     0.  ,     0.  ,
            0.  ,     0.  ,     0.  ,     0.  ]], dtype='float32'), np.array([[0.0000000e+00, 5.3419344e+04, 0.0000000e+00, 0.0000000e+00,
        4.8885941e+01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00]], dtype='float32'), np.array([[    0.   , 50534.137,     0.   ,     0.   ,     0.   ,     0.   ,
            0.   ,     0.   ,     0.   ,     0.   ]], dtype='float32'), np.array([[    0.  , 60471.15,     0.  ,     0.  ,     0.  ,     0.  ,
            0.  ,     0.  ,     0.  ,     0.  ]], dtype='float32'), 
        np.array([[     0. , 125936.4,      0. ,      0. ,      0. ,      0. ,
             0. ,      0. ,      0. ,      0. ]], dtype='float32')])

c=results
p=list()
c=c.reshape(6,10)
for i in range(6):
    p.append(c[i][1])
ref=list();
for i in range(6):
    ref.append(c[0][1])

x=np.asarray(p)
x=x.reshape(6,1)
q=np.array(ref)
q=q.reshape(1,6)
train_X=np.concatenate((x,q.T), axis=1)
b=np.array([0,0,0,0,1,1])
b=b.reshape(6,1)
c=np.append(train_X,b,axis=1)
df=pd.DataFrame(c,columns=['1','2', 'y'])
X_train=df.ix[:,0:2]
y_train=np.ravel(df.y)
print(X_train)
scale=StandardScaler().fit(X_train)
X_train=scale.transform(X_train)

model.add(Dense(10, activation='relu', input_shape=(2,)))

# Add one hidden layer 
model.add(Dense(5, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=100, batch_size=1, verbose=1)

pred=df.ix[5:,0:2]
h=model.predict(pred, verbose=1)



