#from models.feedforward import feedforward
from data_loader import DataLoader

MAX_EPOCHS = 15
learning_rate = 0.001

import tensorflow as tf
import numpy as np

width = 32
height = 32
channels = 3
n_classes = 10


# Define placeholders (inputs and outputs given to model)

x = tf.placeholder(tf.float32, shape = (None, width, height, channels))
y = tf.placeholder(tf.float32, shape = (None, n_classes))

n_hidden_1 = 1024
n_hidden_2 = 1024
n_hidden_3 = 512
n_hidden_4 = 512

#CHECK flat_x = tf.reshape(x, [-1])
flat_x = tf.layers.flatten(x)

#CHECK _, n_input = tf.shape(flat_x)
n_input = width*height*channels 

weights = {
'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
'h3' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
'h4' : tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
'out' : tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}

biases = {
'h1' : tf.Variable(tf.random_normal([n_hidden_1])),
'h2' : tf.Variable(tf.random_normal([n_hidden_2])),
'h3' : tf.Variable(tf.random_normal([n_hidden_3])),
'h4' : tf.Variable(tf.random_normal([n_hidden_4])),
'out' : tf.Variable(tf.random_normal([n_classes]))
}


# Define model (graph)
# Start with feedforward model

layer_1 = tf.nn.relu(tf.add(tf.matmul(flat_x, weights['h1']), biases['h1']))
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['h2']))
layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['h3']))
layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['h4']))
out = tf.add(tf.matmul(layer_4, weights['out']), biases['out'])

logits = out


# Define Cross-Entropy Loss

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))


# Define optimizer
# and the optimize operation

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)


# Define session and initialize weights

init = tf.global_variables_initializer()


# For training, implement the following using DataLoader Object
# Run training MAX_EPOCHS times
# In every epoch, cycle through all training data randomly
# After every epoch, save model weights in folder model_weights using tf.train.Saver()
# After every epoch, compute avg. loss on validation set
# Stop training once validation cost starts increasing
# Split entire training data into train (90%) and val (10%) sets
















# Load weights using tf.train.Saver() from epoch with lowest validation cost.
# Compute avg. test accuracy and loss
