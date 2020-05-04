import tensorflow as tf

from vgg16 import vgg16
from resnet_model import Model

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

def ResNet(x, training):

    resnet_size = 50 # Set resnet_size to be of the form 6n + 2
    num_blocks = (resnet_size - 2) // 6
    resnet = Model(resnet_size=resnet_size,
        bottleneck=False,
        num_classes=n_classes,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        resnet_version=2,
        data_format='channels_last',
        dtype=tf.float32
        )
    return resnet(x, training)

def get_logits(x, training, model_name):

    if model_name == 'feedforward':
        out = feedforward(x)
    elif model_name == 'CNN':
        out = CNN(x)
    elif model_name == 'VGG16':
        vgg = vgg16(x)
        out = vgg.CIFAR_fc_layers()
        return out, vgg
    elif model_name == 'ResNet':
        out = ResNet(x, training)
    return out
