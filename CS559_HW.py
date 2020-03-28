# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:48:40 2020

@author: arda1
"""

from matplotlib.image import imread
import glob
import numpy as np
import tensorflow as tf


batchSize = 100

# Loads data from given path and given batch size
def loadData(path, batchSize = -1):
    
    counter = 0
    images = [] 
    labels = []
    
    if batchSize == -1:
        batchSize = len(glob.glob('./training/*.jpg'))
            
    for filename in glob.glob('./training/*.jpg'): #assuming jpg
        if counter < batchSize:
            # Image related operations
            im=imread(filename)
            im = np.resize(im,(80, 80, 3))
            im = np.true_divide(im, 255)
            images.append(im)
            
            # Label related operations
            label = filename.split("\\")[1].split(".")[0].split("_")[0]
            labels.append(label)
        else: 
            break
        counter += 1
        
    return images, labels


# Applies a single convolution operation on x with parameters w and b and settings stride and pad
# Applies ReLU activation operation
def conv2D(x, W, b, stride, pad = 'SAME'):
    
    x = tf.nn.conv2d(x, W, stride, pad)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    
# Applies max-pool operation on x
def maxpool2d(x, kSize=2, pad = 'SAME'):
    return tf.nn.max_pool(x, kSize, kSize, pad)


# Assuming the data is in same folder with source code
batch, labels = loadData('./training/*.jpg', 100)

    