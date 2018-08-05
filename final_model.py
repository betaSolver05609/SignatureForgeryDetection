import numpy as np
import pandas as pd;
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import glob
import os
np.random.seed(2018)

PATH="C:\\Users\\Dell inspiron\\Desktop\\FinalProject"
class Discriminator(object):
    def __init__(self, data, label):
        self.data=data
        self.label=label
    def prepare_data(self):
        c=self.data
        dim=int(c.shape[0])
        c=c.reshape(dim,10)
        ref=list()
        for i in range(dim):
            ref.append(c[1])
        ref=np.array(ref)
        ref=ref.reshape(10,dim)
        X_train=np.concatenate((c,ref.T), axis=1)
        os.chdir(PATH+'\\ready')
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
        
    def train(self):
        X_train,y_train=self.prepare_data()
        model=Sequential()
        model.add(Dense(30, activation='relu', input_shape=(20,)))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        model.fit(X_train, y_train,epochs=100, batch_size=1, verbose=1)
        return model
    def prepare_test_set(self):
        c=self.label
        dim=int(c.shape[0])
        c=c.reshape(dim,10)
        #keeping reference same
        p=self.data
        dim1=int(p.shape[0])
        p=p.reshape(dim1,10)
        ref=list()
        for i in range(dim):
            ref.append(p[1])
        ref=np.array(ref)
        ref=ref.reshape(dim,10)
        #====================
        X_train=np.concatenate((c,ref), axis=1)
        
        df=pd.DataFrame(X_train)
        X_train=df.ix[:,0:20]
       
        scale=StandardScaler().fit(X_train)
        X_test=scale.transform(X_train)
        return X_test
        
    def predict(self):
        X_train,y_train=self.prepare_data()
        model=self.train()
        X_test=self.prepare_test_set()
        return model.predict(X_test, verbose=1)
    
        