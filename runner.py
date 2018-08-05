
import numpy as np
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
import numpy as np
from keras.callbacks import History


history=History();
a=np.array([[    0., 83049.,     0.,     0.,     0.,     0.,     0.,     0.,
            0.,     0.]], dtype='float32')

b=np.array([[    0.  , 83266.43,     0.  ,     0.  ,     0.  ,     0.  ,
            0.  ,     0.  ,     0.  ,     0.  ]], dtype='float32')


a=np.ravel(a)
b=np.ravel(b)
c=list();
c.append(a[1])
c.append(b[1])

feature=abs(c[0]-c[1])
feature=np.array(feature)
feature.reshape(1,)
model=Sequential();

#Adding input and max-hiehght layer with relu activation function (I though relu would work better than sigmoid so used it)
model.add(Dense(20, activation='relu', input_shape=(1,)))

#adding another layer. See documentaion for further details
model.add(Dense(12, activation='relu'))

model.add(Dense(1, activation='sigmoid'));

#Refer to documentation for explanation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

y=np.array([1]).reshape(1,)
model.fit([feature], y, batch_size=1, epochs=10, verbose=1, callbacks=[history])




