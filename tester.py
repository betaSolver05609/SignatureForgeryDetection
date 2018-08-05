
#from Classifier import ClassifiersDesign
#from Classifier import ClassifierDesign
from scipy import misc
import numpy as np
import pandas as pd
import os
import glob
from final_model import Discriminator
np.random.seed(2018)
img=misc.imread('sample23.png')
PATH="C:\\Users\\Dell inspiron\\Desktop\\FinalProject"
#Tester for Discriminator
"""results=([np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.  , 50029.36,     0.  ,     0.  ,     0.  ,     0.  ,
            0.  ,     0.  ,     0.  ,     0.  ]], dtype='float32'), np.array([[0.0000000e+00, 5.3419344e+04, 0.0000000e+00, 0.0000000e+00,
        4.8885941e+01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00]], dtype='float32'), np.array([[    0.   , 50534.137,     0.   ,     0.   ,     0.   ,     0.   ,
            0.   ,     0.   ,     0.   ,     0.   ]], dtype='float32'), np.array([[    0.  , 60471.15,     0.  ,     0.  ,     0.  ,     0.  ,
            0.  ,     0.  ,     0.  ,     0.  ]], dtype='float32'), 
        np.array([[     0. , 125936.4,      0. ,      0. ,      0. ,      0. ,
             0. ,      0. ,      0. ,      0. ]], dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32'), np.array([[    0.     , 25411.39   ,     0.     ,     0.     ,   199.69936,
            0.     ,     0.     ,     0.     ,     0.     ,     0.     ]],dtype='float32')])
results=np.array(results)

obj=Discriminator(results, results)

a=obj.predict()
"""


#Tester for Classifier
"""
obj=ClassifierDesign(img,1)
obj.train()
x=obj.main_conv()
#a,b,c,d=obj.prepare_data()
"""

