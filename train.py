#TODO: Get model from config file
#      Add progress bar
#      Reset the placeholders and values before each iteration just in case

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from PIL import Image
import numpy as np
import random
import time
import json

def parsecfg(filedir):
    file = open(filedir, 'r')
    lines = file.read().split('\n')                        
    lines = [x for x in lines if len(x) > 0]               
    lines = [x for x in lines if x[0] != '#']              
    lines = [x.rstrip().lstrip() for x in lines]           

    block = {}
    blocks = []

    for line in lines:
      if line[0] == "[":               
        if len(block) != 0:         
          blocks.append(block)     
          block = {}               
        block["type"] = line[1:-1].rstrip()    
      else:
        key,value = line.split("=")
        block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks

blocks=parsecfg("HWNet.txt")
hyperparameters = {}
for key in blocks[0]:
  val = blocks[0][key]
  hyperparameters[key]=val
print(hyperparameters)

def data_loader(file_dir, num_data, size, channel_type):
    print("Started loading data")
    images = []
    labels = []
    count  = 0
    
    all_files = os.listdir(file_dir)

    r = list(range(num_data))
    random.shuffle(r)
    for i in r:
        if channel_type =="greyscale":
          im = Image.open(file_dir+all_files[i],"r").convert('L') #getting the image, converting it to the greyscale
        else:
          im = Image.open(file_dir + all_files[i],"r")            #getting the image
        label = all_files[i].split(".jpg")[0].split("_")[0]       #getting the label from name
        temp_data = np.asarray(im.getdata())                      #converting the image -> pixels
        pix_val = np.resize(temp_data,(size, size, 1))            #the data already in expected size but just in case
        pix_val = np.true_divide(pix_val, 255)                    #normalization of pixels
        images = np.append(images, pix_val)                       #putting all pixels to one array, will be reshaped later
        labels = np.append(labels, label)                         #putting all labels to one array, will be reshaped later 
        count = count + 1                                         #counting the number of samples


    images = np.reshape(images, (count, size,size,1))             #pixels reshaped
    images = images.astype('float32')
    labels = np.reshape(labels, (count, 1))                       #labels reshaped
    labels = labels.astype('float32')

    print("shape for images: " + str(images.shape))               #just to make sure everything is retrieved as we wanted
    print("shape for labels: " + str(labels.shape))             

    return images, labels



print("Getting the training data")
# file_dir_train = "/content/gdrive/My Drive/CS559/SCUT_FBP5500_downsampled/training/"
file_dir_train = "./training/"
images, labels = data_loader(file_dir_train,3550, int(hyperparameters['img_size']) , hyperparameters['train_color']) #3550
print(hyperparameters['train_color'])
if hyperparameters['train_color']=='greyscale':
  num_channels= 1
elif hyperparameters['train_color']=='rgb':
  num_channels = 3

print("Getting the testing data")
# file_dir_val = "/content/gdrive/My Drive/CS559/SCUT_FBP5500_downsampled/test/"
file_dir_val = "./test/"
images_val, labels_val = data_loader(file_dir_val, 890, int(hyperparameters['img_size']), hyperparameters['train_color']) #892

#Creating the conv layer for the network
#if the net part of the config file has 1 for batch normalization we activate the batch normalization here
def conv2D(x, W, b, stride = 1, pad = 'SAME'):
    x = tf.nn.conv2d(x, W, stride, pad)
    x = tf.nn.bias_add(x, b)
    
    if hyperparameters['batch_normalization'] == 1:
      mean, var = tf.nn.moments(x, [0,1,2], name='moments')
      x = tf.nn.batch_normalization(x, mean, var, variance_epsilon = 1e-3, name=None)
      # x = tf.compat.v1.layers.batch_normalization(x, training = True)
    x = tf.nn.relu(x)
    
    return  x


def maxpool2d(x, k):
    ksize = [1,k,k,1]
    strides= [1,k,k,1]
    return tf.nn.max_pool2d(x, ksize, strides,padding='SAME')

#we need placeholders to pass data in when the session will be run.
def create_placeholders(n_s, n_xc, n_y):
    X = tf.placeholder(tf.float32, [None, n_s, n_s, n_xc], name="X")
    Y = tf.placeholder(tf.float32, [None, n_y], name="Y")    
    return X, Y


def initialize_parameters(blocks):
  parameters_w = {}
  parameters_b = {}
  num_filter = []
  init= hyperparameters['init']
  initializer =None
  if init =='xavier':
    print(init + " initialized")
    initializer= tf.contrib.layers.xavier_initializer()
  elif init == 'gauss':
    print(init + " initialized")
    initializer = tf.compat.v1.random_uniform_initializer()

  with tf.compat.v1.variable_scope("init",reuse=tf.compat.v1.AUTO_REUSE):
    for i in range(len(blocks)):
      module_type = blocks[i]['type']
      if i !=0:
        w_name = 'w'+str(i)
        b_name = 'b'+str(i)
        stride = bool(blocks[i]['stride'])
        kernel = int(blocks[i]['kernel'])
        num_filters = int(blocks[i]['filter'])
        num_filter = np.append(num_filter, num_filters)
        if module_type == 'conv':
          if i ==1:
            parameters_w[i] = tf.compat.v1.get_variable(w_name, shape = [kernel, kernel, num_channels, num_filters]   , initializer=initializer)
            parameters_b[i] = tf.compat.v1.get_variable(b_name ,shape = (num_filters) , initializer = initializer)
          else:
            parameters_w[i] = tf.compat.v1.get_variable(w_name, shape = [kernel, kernel, int(num_filter[i-2]), int(num_filters)]   , initializer = initializer)
            parameters_b[i] = tf.compat.v1.get_variable(b_name ,shape = (num_filters) , initializer = initializer)
        elif module_type == 'fully':
          #calculate the final shape according to number of strides and write for the shape
          parameters_w[i] = tf.compat.v1.get_variable(w_name , shape = [(5*5*num_filters), num_filters], initializer = initializer)
          parameters_b[i] = tf.compat.v1.get_variable('bd1', shape = (num_filters), initializer = initializer)
        elif module_type == 'regression':
          parameters_w[i] = tf.compat.v1.get_variable(w_name , shape = [num_filters, 1] , initializer = initializer)
          parameters_b[i] = tf.compat.v1.get_variable(b_name , shape = (1), initializer = initializer)
    return parameters_w, parameters_b 

parameters_w, parameters_b = initialize_parameters(blocks)

def forward_propagation(X, p_w, p_b):    
    layers = {}
    for i, module in enumerate(blocks):
      module_type = (module['type'])
      if module_type == 'conv':
          
        stride = int(blocks[i]['stride'])
        layer_name = str(i)
        if bool(layers) == False:
          layers[layer_name] = conv2D(X, p_w[i] , p_b[i])
          layers[layer_name] = maxpool2d(layers[str(i)], k=stride)
          print(str(layers[layer_name].shape))
        else:
          layers[layer_name] = conv2D(layers[str(i-1)], p_w[i] , p_b[i])
          layers[layer_name] = maxpool2d(layers[str(i)], k=stride)
          print(str(layers[layer_name].shape))
      elif module_type == 'fully':
        #Fully connecetd
        layer_name=str(i)
        layers[layer_name] = tf.reshape(layers[str(i-1)], [-1, parameters_w[i].get_shape().as_list()[0]])
        layers[layer_name] = tf.add(tf.matmul(layers[layer_name], parameters_w[i]), parameters_b[i])
        
        if hyperparameters['batch_normalization'] == 1:
           mean, var = tf.nn.moments(layers[layer_name], [0], name='moments')
           layers[layer_name] = tf.nn.batch_normalization(layers[layer_name], mean, var, variance_epsilon = 1e-3, name=None)
       
        layers[layer_name] = tf.nn.relu(layers[str(i)])
        
        if hyperparameters['dropout'] == 1:
            layers[layer_name] = tf.nn.dropout(layers[layer_name], 0.5)
            
        print(layers[layer_name].shape)
      elif module_type == 'regression':
        print(p_w[i])
        reg_layer = tf.add(tf.matmul(layers[str(i-1)], p_w[i]), p_b[i])
        print(reg_layer.shape)
        return reg_layer
  
    return "Error: There is no regression layer. Add regression layer to the cfg file"

def getLoss(loss, Y, pred):
    if loss == "mse":
        return tf.compat.v1.losses.mean_squared_error(labels=Y, predictions=pred);
    elif loss == "mae":
        return tf.reduce_mean(tf.abs(tf.subtract(Y, pred)))
    elif loss == "logcosh":
        return tf.math.reduce_sum(tf.math.log(tf.cosh(pred - Y)))

X = tf.placeholder("float")
Y = tf.placeholder("float")

n_samples = images.shape[0]

# blocks=parsecfg("/content/gdrive/My Drive/CS559/HWNet.cfg")

parameters_w, parameters_b = initialize_parameters(blocks)
pred = forward_propagation(X, parameters_w, parameters_b)
loss = getLoss(hyperparameters['loss'], Y, pred)
regularizers = tf.nn.l2_loss(parameters_w[1]) + tf.nn.l2_loss(parameters_w[2]) + tf.nn.l2_loss(parameters_w[3]) + tf.nn.l2_loss(parameters_w[4]) + \
                    tf.nn.l2_loss(parameters_w[5]) + tf.nn.l2_loss(parameters_w[6]) + \
                    tf.nn.l2_loss(parameters_w[7]) + tf.nn.l2_loss(parameters_w[8])
loss = loss + float(hyperparameters['beta']) * regularizers
with tf.compat.v1.variable_scope("opti",reuse=tf.compat.v1.AUTO_REUSE):
    optimizer = tf.compat.v1.train.AdamOptimizer(float(hyperparameters['learning_rate']), float(hyperparameters['beta1']), float(hyperparameters['beta2']), float(hyperparameters['epsilon'])).minimize(loss)

# Round predictions for MAE calculation
predEr = tf.math.round(pred)
error = tf.compat.v1.metrics.mean_absolute_error(Y, predictions=predEr)

batch_size=int(hyperparameters['batch_size'])
epoch= int(hyperparameters['epoch'])
init = tf.compat.v1.global_variables_initializer()
# with tf.device("/device:GPU:0"):
with tf.compat.v1.Session() as sess:
    print("Session started")
    sess.run(init)
    sess.run(tf.compat.v1.local_variables_initializer())
    val_X = images_val
    val_Y = labels_val
    
    loss_epoch=[]
    
    for ithepoch in range(int(epoch)):
        #tic = time.perf_counter()
        for batch in range(len(images)//batch_size):
            batch_x = images[batch*batch_size:min((batch+1)*batch_size,len(images))]
            #print(batch_x.shape)
            batch_y = labels[batch*batch_size:min((batch+1)*batch_size,len(images))]  
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
            loss_train = sess.run(loss, feed_dict={X: batch_x, Y:batch_y})
            #toc = time.perf_counter()
            #print(f"\trun one batch is in {toc - tic:0.4f} seconds")
        #err_train = sess.run(error, feed_dict={X: batch_x, Y:batch_y})
        
        #make predictions integer
        loss_epoch=np.append(loss_epoch,loss_train)

        print("Epoch:", '%04d' % (ithepoch+1), "loss=",loss_train)
        #
        
        loss_val = sess.run(loss, feed_dict={X:images_val, Y:labels_val})
        err_val = sess.run(error,feed_dict={X:images_val, Y:labels_val})
        print("            loss_val=",loss_val, "error_val=",err_val[0])
        print()
