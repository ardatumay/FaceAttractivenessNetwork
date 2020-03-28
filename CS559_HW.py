# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:48:40 2020

@author: arda1
"""

from matplotlib.image import imread
import glob

image_list = []
batch_size = 100

counter = 0
for filename in glob.glob('./training/*.jpg'): #assuming jpg
    if counter < batch_size:
        im=imread(filename)
        image_list.append(im)
    else: 
        break
    counter += 1