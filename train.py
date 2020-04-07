#TODO: Get model from config file
#      Add progress bar
#      Reset the placeholders and values before each iteration just in case

import tensorflow as tf
from data_loader import data_loader
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)
tf.disable_v2_behavior()

<<<<<<< HEAD
file_dir_train = "SCUT_FBP5500_downsampled/training/"
images, labels = data_loader(file_dir_train,100) #3550

file_dir_val = "SCUT_FBP5500_downsampled/test/"
images_val, labels_val = data_loader(file_dir_val, 100) #892
=======
print(tf.__version__)
tf.disable_v2_behavior()

file_dir_train = "SCUT_FBP5500_downsampled/training/"
images, labels = data_loader(file_dir_train)

file_dir_test = "SCUT_FBP5500_downsampled/test/"
images_test, labels_test = data_loader(file_dir_test)
>>>>>>> 8fd0c606fb41179d0b8eabeb67d0ec0c9c5eabae

training_iters = 10 
learning_rate = 0.001 
batch_size = 2
n_input = 80
n_classes = 8

<<<<<<< HEAD
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def conv2d(x, W, b, strides=1, pad):
=======
def conv2d(x, W, b, strides=1):
>>>>>>> 8fd0c606fb41179d0b8eabeb67d0ec0c9c5eabae
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=pad)
    x = tf.nn.bias_add(x, b)
    x = tf.compat.v1.layers.batch_normalization(x, training = True)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
<<<<<<< HEAD
    x = tf.nn.dropout( x, 0.5)
=======
>>>>>>> 8fd0c606fb41179d0b8eabeb67d0ec0c9c5eabae
    return tf.nn.max_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

#to later pass your training data in when you run your session.
def create_placeholders(n_s, n_xc, n_y):
    X = tf.placeholder(tf.float32, [None, n_s, n_s, n_xc], name="X")
    Y = tf.placeholder(tf.float32, [None, n_y], name="Y")    
    return X, Y

def initialize_parameters():
<<<<<<< HEAD
    with tf.compat.v1.variable_scope("init",reuse=tf.compat.v1.AUTO_REUSE):
        W0  = tf.compat.v1.get_variable('wc0' , shape = [3, 3, 3, 32]   , initializer = tf.contrib.layers.xavier_initializer())
        W1  = tf.compat.v1.get_variable('wc1' , shape = [3, 3, 32, 64]  , initializer = tf.contrib.layers.xavier_initializer())
        W2  = tf.compat.v1.get_variable('wc2' , shape = [3, 3, 64, 128] , initializer = tf.contrib.layers.xavier_initializer())
        WD1 = tf.compat.v1.get_variable('wd1' , shape = [10*10*128, 128], initializer = tf.contrib.layers.xavier_initializer())
        out = tf.compat.v1.get_variable('out' , shape = [128, 1] , initializer = tf.contrib.layers.xavier_initializer())
        
        b0  = tf.compat.v1.get_variable('b0' , shape = (32) , initializer = tf.contrib.layers.xavier_initializer())
        b1  = tf.compat.v1.get_variable('b1' , shape = (64) , initializer = tf.contrib.layers.xavier_initializer())
        b2  = tf.compat.v1.get_variable('b2' , shape = (128), initializer = tf.contrib.layers.xavier_initializer())
        bd1 = tf.compat.v1.get_variable('bd1', shape = (128), initializer = tf.contrib.layers.xavier_initializer())
        b_o = tf.compat.v1.get_variable('bo' , shape = (1), initializer = tf.contrib.layers.xavier_initializer())
=======
    print("initialize_parameters")
    with tf.compat.v1.variable_scope("init",reuse=tf.compat.v1.AUTO_REUSE):
        W0  = tf.compat.v1.get_variable('wc0' , shape = [3, 3, 3, 32])
        W1  = tf.compat.v1.get_variable('wc1' , shape = [3, 3, 32, 64])
        W2  = tf.compat.v1.get_variable('wc2' , shape = [3, 3, 64, 128])
        WD1 = tf.compat.v1.get_variable('wd1' , shape = [10*10*128, 128])
        out = tf.compat.v1.get_variable('out', shape=[128, 1])
        
        b0  = tf.compat.v1.get_variable('b0' , shape = (32) )
        b1  = tf.compat.v1.get_variable('b1' , shape = (64) )
        b2  = tf.compat.v1.get_variable('b2' , shape = (128))
        bd1 = tf.compat.v1.get_variable('bd1', shape = (128))
        b_o = tf.compat.v1.get_variable('bo' , shape = (1))
>>>>>>> 8fd0c606fb41179d0b8eabeb67d0ec0c9c5eabae
    

        parameters = {"W0" : W0, "b0" : b0,
                      "W1" : W1, "b1" : b1,
                      "W2" : W2, "b2" : b2,
                      "WD1": WD1,"bd1": bd1,
                      "out": out,"b_o": b_o
                      }
    return parameters 

def forward_propagation(X, parameters):
<<<<<<< HEAD
=======
    print("forward_prop")
>>>>>>> 8fd0c606fb41179d0b8eabeb67d0ec0c9c5eabae
    W0  = parameters['W0']
    b0  = parameters['b0']
    W1  = parameters['W1']
    b1  = parameters['b1']
    W2  = parameters['W2']
    b2  = parameters['b2']
    WD1 = parameters['WD1']
    bd1 = parameters['bd1']
    o = parameters['out']
    b_o = parameters['b_o']
    
<<<<<<< HEAD
    conv1 = conv2d(X, W0, b0, 'SAME')
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, W1, b1, 'SAME')
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, W2, b2, 'SAME')
=======
    conv1 = conv2d(X, W0, b0)
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, W1, b1)
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, W2, b2)
>>>>>>> 8fd0c606fb41179d0b8eabeb67d0ec0c9c5eabae
    conv3 = maxpool2d(conv3, k=2)
    
    fc1 = tf.reshape(conv3, [-1, WD1.get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, WD1), bd1)
    fc1 = tf.nn.relu(fc1)
    
    outL = tf.add(tf.matmul(fc1, o), b_o)
    print(outL.shape)
<<<<<<< HEAD
    print(outL)
=======
>>>>>>> 8fd0c606fb41179d0b8eabeb67d0ec0c9c5eabae
    
    return outL

X = tf.placeholder("float")
Y = tf.placeholder("float")
<<<<<<< HEAD

n_samples = images.shape[0]
beta = 0.1

parameters = initialize_parameters()
pred = forward_propagation(images, parameters)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
regularizers = tf.nn.l2_loss(parameters['W0']) + tf.nn.l2_loss(parameters['W1']) + \
                   tf.nn.l2_loss(parameters['W2']) + tf.nn.l2_loss(parameters['WD1'])
loss = tf.reduce_mean(loss + beta * regularizers)
with tf.compat.v1.variable_scope("opti",reuse=tf.compat.v1.AUTO_REUSE):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

batch_size=2

init = tf.compat.v1.global_variables_initializer()
=======
n_samples = images.shape[0]

parameters = initialize_parameters()
pred = forward_propagation(images, parameters)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.compat.v1.global_variables_initializer()

>>>>>>> 8fd0c606fb41179d0b8eabeb67d0ec0c9c5eabae
with tf.compat.v1.Session() as sess:
    sess.run(init) 
    pred = pred.eval() 
    summary_writer = tf.compat.v1.summary.FileWriter('./Output', sess.graph)
<<<<<<< HEAD
    for epoch in range(40):
        for batch in range(len(images)//batch_size):
            batch_x = images[batch*batch_size:min((batch+1)*batch_size,len(images))]
            batch_y = labels[batch*batch_size:min((batch+1)*batch_size,len(images))]  
            opt= sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        l = sess.run(loss, feed_dict={X: pred, Y:batch_y})
        print("Epoch:", '%04d' % (epoch), "loss=", "{:.9f}".format(l))
        print("Optimization Finished!")
            
    test_X = images_val
    test_Y = labels_val

    print("Testing... (Mean square loss Comparison)")
    testing_loss = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing loss=", testing_loss)
    testing_mae = tf.compat.v1.metrics.mean_absolute_error(test_Y, pred)
=======
    for epoch in range(100):
        for (x, y) in zip(images, labels):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            if (epoch+1) % 50 == 0:
                c = sess.run(cost, feed_dict={X: pred, Y:y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: images, Y: labels})
        print("Training cost=", training_cost)
    summary_writer.close()
>>>>>>> 8fd0c606fb41179d0b8eabeb67d0ec0c9c5eabae
