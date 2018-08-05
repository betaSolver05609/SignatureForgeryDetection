# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 01:29:33 2017

@author:SonuHacker
"""
"""This module defines a class which defines the blueprint for preprocessing
of an Image. The image is the DataMember and several methods as described
are Encapsulated. Thus the preprocessing has a black box behavior
"""

import numpy as np;
from scipy import ndimage,misc

class PreprocessingEngine(object):
    #DataMembers of class:
        #Image
    #Methods of class    
        #GrayScale Conversion
        #Scaling
        #NoiseReduction
        #BackgroundElimination
        #SignatureNormalization
        #Thinning

    def __init__(self,image):
        self.image=image;
    #Converting traditional image into gray Image
    #Using formula image array dot [0.299R, 0.587G, 0144B]
    def getGrayImage(self):
        self.image=np.dot(self.image[...,:3], [0.299, 0.587, 0.114])
        return self.image
    #Scaling
    def ScaleImage(self):
        #x(new)=x(old)*100/H
        #y(new)=y(old)*100/W
        self.image=misc.imresize(self.image,(105,105));
        return self.image;
    
    #NoiseReduction
    def reduceNoise(self):
        #median filter presrves edges better, so gaussian filter is not used
        self.image=ndimage.median_filter(self.image,5)
        return self.image;
    #BackgroundElimination
    def eliminateBackground(self):
        thresh=self.image.mean()
        #Thresholding is used to eliminate background
        #if T is threshold( here it is thresh)
        #f(x,y)>=T:- foreground/obj
        #f(x,y)<T:- Background
        binary=self.image>thresh 
        #returned Image is a binary file
        self.image=binary;        
        return binary;
    
