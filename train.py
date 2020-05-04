import sys
import tensorflow as tf

from data_loader import DataLoader
from models import get_logits


MAX_EPOCHS = 30
learning_rate = 0.003
batch_size = 32
decay_constant = 0.50

width = 32
height = 32
channels = 3
n_classes = 10


# Define placeholders (inputs and outputs given to model)

x = tf.placeholder(tf.float32, shape=(None, height, width, channels))
y = tf.placeholder(tf.float32, shape=(None, n_classes))
lr = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

if sys.argv[1] == 'VGG16':
    logits, vgg = get_logits(x, training, sys.argv[1])
else:
    logits = get_logits(x, training, sys.argv[1])

# Define Cross-Entropy Loss

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))


# Apply softmax to logits

pred = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))


# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Define optimizer
# and the optimize operation

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = optimizer.minimize(loss_op)
train_op = tf.group([train_op, update_ops])


#Create Object of class DataLoader

data_loader = DataLoader()

# Define session and initialize weights
# For training, implement the following using DataLoader Object
# Run training MAX_EPOCHS times
# In every epoch, cycle through all training data randomly
# After every epoch, save model weights in folder model_weights using tf.train.Saver()
# After every epoch, compute avg. loss on validation set


saver = tf.train.Saver(max_to_keep=100)
init = tf.global_variables_initializer()

min_val_cost = float('inf')
min_cost_ind = -1

with tf.Session() as sess:
    sess.run(init)

    if sys.argv[1] == 'VGG16':
        vgg.load_weights('vgg16_weights.npz', sess)

    for epoch in range(MAX_EPOCHS):
        data_loader.reset_epoch()
        avg_cost = 0.
        n_batches = data_loader.num_batches(batch_size, 'train')

        for i in range(n_batches):
            batch_x, batch_y = data_loader.get_batch(batch_size, 'train')
            _, c = sess.run([train_op, loss_op], feed_dict={x: batch_x, y: batch_y, training: True, lr: learning_rate})
            avg_cost += c / n_batches


        # Validation Cost Calculated in batches
        # Increase n_batches_val in case of run out of memory errors

        val_cost = 0.
        n_batches_val = 5
        batch_size_val = int(data_loader.n_val / n_batches_val)

        for i in range(n_batches_val):   #Loop to Calculate Validation Cost
            batch_x, batch_y = data_loader.get_batch(batch_size_val, 'val')
            c = loss_op.eval({x: batch_x, y: batch_y, training: False})
            val_cost += c / n_batches_val

        print("Epoch:", '%04d' % (epoch+1), "Training cost = {:.9f}".format(avg_cost), "Validation cost = {:.9f}".format(val_cost))

        if val_cost < min_val_cost:
            min_val_cost = val_cost
            min_cost_ind = (epoch+1)
            saver.save(sess, './weights/epoch', global_step=(epoch+1))

        if (epoch+1) % 10 == 0:
            learning_rate = learning_rate * decay_constant
            print("New Learning Rate after decay is", learning_rate)

    # Test Accuracy
    # Load weights using tf.train.Saver() from epoch with lowest validation cost.
    # Compute avg. test accuracy str(min_cost_ind)

    saver.restore(sess, './weights/epoch-' + str(min_cost_ind) )

    # Average Test Accuracy Calculated in batches
    # Increase n_batches_test in case of run out of memory errors

    avg_accuracy = 0.
    n_batches_test = 10
    batch_size_test = int(data_loader.n_test / n_batches_test)

    for i in range(n_batches_test):   #Loop to Calculate average Test Accuracy
        batch_x, batch_y = data_loader.get_batch(batch_size_test, 'test')
        c = accuracy.eval({x: batch_x, y: batch_y, training: False})
        avg_accuracy += c / n_batches_test

    print("Minimum Validation Cost is", min_val_cost, "on epoch", min_cost_ind)
    print("Test Accuracy on epoch", min_cost_ind, "is", avg_accuracy)
