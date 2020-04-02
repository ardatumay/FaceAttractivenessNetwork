import tensorflow as tf
from data_loader import data_loader

print(tf.__version__)
tf.disable_v2_behavior()

images, labels = data_loader()
images= tf.convert_to_tensor(images, dtype='float32', name='X_train')
labels= tf.convert_to_tensor(labels, dtype='float32', name='Y_train')

training_iters = 10 
learning_rate = 0.001 
batch_size = 128
n_input = 80
n_classes = 8

x = tf.placeholder("float", [None, 80,80,3])
y = tf.placeholder("float", [None, n_classes])


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

#to later pass your training data in when you run your session.
def create_placeholders(n_s, n_xc, n_y):
    X = tf.placeholder(tf.float32, [None, n_s, n_s, n_xc], name="X")
    Y = tf.placeholder(tf.float32, [None, n_y], name="Y")    
    return X, Y

def initialize_parameters():
    print("initialize_parameters")
    with tf.compat.v1.variable_scope("init",reuse=tf.compat.v1.AUTO_REUSE):
        W0  = tf.get_variable('wc0' , shape = [3, 3, 3, 32])
        W1  = tf.get_variable('wc1' , shape = [3, 3, 32, 64])
        W2  = tf.get_variable('wc2' , shape = [3, 3, 64, 128])
        WD1 = tf.get_variable('wd1' , shape = [10*10*128, 128])
        out = tf.get_variable('out', shape=[128, 1])
        
        b0  = tf.get_variable('b0' , shape = (32) )
        b1  = tf.get_variable('b1' , shape = (64) )
        b2  = tf.get_variable('b2' , shape = (128))
        bd1 = tf.get_variable('bd1', shape = (128))
        b_o = tf.get_variable('bo' , shape = (1))
    

        parameters = {"W0" : W0, "b0" : b0,
                      "W1" : W1, "b1" : b1,
                      "W2" : W2, "b2" : b2,
                      "WD1": WD1,"bd1": bd1,
                      "out": out,"b_o": b_o
                      }
    return parameters 

def forward_propagation(X, parameters):
    print("forward_prop")
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
    
    conv1 = conv2d(X, W0, b0)
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, W1, b1)
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, W2, b2)
    conv3 = maxpool2d(conv3, k=2)
    
    fc1 = tf.reshape(conv3, [-1, WD1.get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, WD1), bd1)
    fc1 = tf.nn.relu(fc1)
    print(fc1.shape)
    
    outL = tf.add(tf.matmul(fc1, o), b_o)
    print(outL.shape)
    
    return outL

parameters = initialize_parameters()
deneme = forward_propagation(images, parameters)

def compute_cost(Z3, Y):    
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

tf.reset_default_graph()

###Bookmark
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(images, parameters)
    cost = compute_cost(Z3, labels)
    print("cost = " + str(cost))
