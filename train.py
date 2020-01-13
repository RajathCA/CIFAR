#from models.feedforward import feedforward
from data_loader import DataLoader

MAX_EPOCHS = 50
learning_rate = 0.001
batch_size = 1024
delay = 3


import tensorflow as tf
import numpy as np

width = 32
height = 32
channels = 3
n_classes = 10


# Define placeholders (inputs and outputs given to model)

x = tf.placeholder(tf.float32, shape = (None, height, width, channels))
y = tf.placeholder(tf.float32, shape = (None, n_classes))

n_hidden_1 = 2048
n_hidden_2 = 2048
n_hidden_3 = 1024
n_hidden_4 = 1024

flat_x = tf.layers.flatten(x)

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


# Apply softmax to logits
pred = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Define optimizer
# and the optimize operation

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)


#Create Object of class DataLoader

data_loader = DataLoader()


# Define session and initialize weights

saver = tf.train.Saver()
init = tf.global_variables_initializer()

val_cost_min = float ('inf')
min_cost_ind = -1
count = 0

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(MAX_EPOCHS):   
        data_loader.reset_epoch() 
        avg_cost = 0.
        n_batches = data_loader.num_batches(batch_size, 'train')

        for i in range(n_batches):
            batch_x, batch_y = data_loader.get_batch(batch_size, 'train')
            _, c = sess.run([train_op, loss_op], feed_dict={x : batch_x, y : batch_y})
            avg_cost += c / n_batches

        batch_x, batch_y = data_loader.get_batch(data_loader.n_val, 'val')
        val_cost = loss_op.eval({x : batch_x, y : batch_y})  #Calculating Validation Cost

        print("Epoch:", '%04d' % (epoch+1), "Training cost = {:.9f}".format(avg_cost), "Validation cost = {:.9f}".format(val_cost))


        if val_cost < val_cost_min:
            val_cost_min = val_cost
            min_cost_ind = (epoch+1)
            count = 0
        
        else:
            count += 1

        if count == delay:
            break

        saver.save(sess, './weights/feed_forward', global_step = (epoch+1))

    #Test Accuracy
    saver.restore(sess, './weights/feed_forward-' + str(min_cost_ind))
    batch_x, batch_y = data_loader.get_batch(data_loader.n_test , 'test')
    print("Accuracy:", accuracy.eval({x : batch_x, y : batch_y}))


# For training, implement the following using DataLoader Object
# Run training MAX_EPOCHS times
# In every epoch, cycle through all training data randomly
# After every epoch, save model weights in folder model_weights using tf.train.Saver()
# After every epoch, compute avg. loss on validation set
# Stop training once validation cost starts increasing
# Split entire training data into train (90%) and val (10%) sets
















# Load weights using tf.train.Saver() from epoch with lowest validation cost.
# Compute avg. test accuracy and loss