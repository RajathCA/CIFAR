import tensorflow as tf

from vgg16 import vgg16

n_classes = 10

def feedforward(x):

    n_hidden_1 = 512
    n_hidden_2 = 512

    flat_x = tf.layers.flatten(x)

	# Define feedforward model (graph)

    layer_1 = tf.layers.dense(flat_x, n_hidden_1, activation=tf.nn.relu)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.relu)
    out = tf.layers.dense(layer_2, n_classes)
    return out

def CNN(x):

	# Define CNN model (graph)
    conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    conv3 = tf.layers.conv2d(conv2, 84, 3, activation=tf.nn.relu)
    conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

    fc1 = tf.layers.flatten(conv3)
    fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 512, activation=tf.nn.relu)
    out = tf.layers.dense(fc2, n_classes)
    return out

def get_logits(x, model_name):

    if model_name == 'feedforward':
        out = feedforward(x)
    elif model_name == 'CNN':
        out = CNN(x)
    elif model_name == 'VGG16':
        vgg = vgg16(x)
        out = vgg.CIFAR_fc_layers()
        return out, vgg
    return out
