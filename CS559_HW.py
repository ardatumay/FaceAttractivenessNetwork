# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:48:40 2020

@author: arda1
"""

from matplotlib.image import imread
import glob
import numpy as np
import tensorflow as tf



# Loads data from given path and given batch size
def loadData(path, batchSize = -1, batchNumber = -1):
    
    counter = 0
    images = [] 
    labels = []
    
    if batchSize == -1 and batchNumber == -1:
        batchSize = len(glob.glob(path))
            
    for filename in glob.glob(path): #assuming jpg
        if counter < batchSize:
            # Image related operations
            im = imread(filename)
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
def conv2D(x, W, b, stride = 1, pad = 'SAME'):
    
    x = tf.nn.conv2d(x, W, stride, pad)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    
# Applies max-pool operation on x
def maxpool2D(x, kSize=2, pad = 'SAME'):
    return tf.nn.max_pool(x, kSize, kSize, pad)

# Applies max-pool operation on x
def avgpool2D(x, kSize=2, pad = 'SAME'):
    return tf.nn.avg_pool(x, kSize, kSize, pad)

# Convolutional network created below
def conv_net(x, weights, biases):  
    
    conv1 = conv2D(x, weights['weight_conv1'], biases['bias_conv1'])
    maxPool1 = maxpool2D(conv1)

    conv2 = conv2D(maxPool1, weights['weight_conv2'], biases['bias_conv2'])
    maxPool2 = maxpool2D(conv2)

    conv3 = conv2D(maxPool2, weights['weight_conv3'], biases['bias_conv3'])
    maxPool3 = maxpool2D(conv3)

    fc1 = tf.reshape(maxPool3, [-1, weights['weight_dense1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['weight_dense1']), biases['bias_dense1'])
    fc1 = tf.nn.relu(fc1)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    
# Relative paths to dataset
trainingPath = "./training/*.jpg"
validationPath = "./validation/*.jpg"
testPath =   "./test/*.jpg"

# Hyperparameters
numEpochs = 100
learning_rate = 0.001 
batchSize = 128

# Data related parameters
inputWidth, inputHeight = 80, 80
numChannels = 3
numClasses = 8

# Placeholders for input images and their labels
x = tf.placeholder("float", [None, inputWidth,inputHeight,numChannels])
y = tf.placeholder("float", [None, numClasses])

weights = {
    'weight_conv1': tf.get_variable('W0', shape=(3, 3, 3, 32), initializer = tf.contrib.layers.xavier_initializer()), 
    'weight_conv2': tf.get_variable('W1', shape=(3, 3, 32, 64), initializer = tf.contrib.layers.xavier_initializer()), 
    'weight_conv3': tf.get_variable('W2', shape=(3, 3, 64, 128), initializer = tf.contrib.layers.xavier_initializer()), 
    'weight_dense1': tf.get_variable('W3', shape=(7*7*128, 128), initializer = tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W4', shape=(128, 1), initializer = tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bias_conv1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bias_conv2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bias_conv3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bias_dense1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(1), initializer=tf.contrib.layers.xavier_initializer()),
}

predictions = conv_net(x, weights, biases)

## SHOULD BE CHANGED ACCORDING TO REGRESSION PROBLEM
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))

## SHOULD BE CHANGED ACCORDING TO REGRESSION PROBLEM
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    # Initialise tf variables
    sess.run(init) 
    
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)

    for i in range(numEpochs):
        for batch in range(len(glob.glob(trainingPath)) // batchSize):
            
             # Assuming the data is in same folder with source code
             train_x, train_y = loadData(trainingPath, batchSize, batch)
    
#            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
#            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
#            # Run optimization op (backprop).
#                # Calculate batch loss and accuracy
#            opt = sess.run(optimizer, feed_dict={x: batch_x,
#                                                              y: batch_y})
#            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
#                                                              y: batch_y})
   











    