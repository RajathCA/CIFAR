import numpy as np

train_percent = 0.9
val_percent = 1 - train_percent
num_channels = 3
img_size = 32
flip_probability = 0.5

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_img_and_label(file):
    dict = unpickle(file)
    raw_img = dict[b'data']
    raw_float = np.array(raw_img, dtype=np.float32) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    labels = dict[b'labels']
    return images, labels

def simultaneous_shuffle(images, labels):
    num_images = images.shape[0]
    shuffle_ind = np.arange(num_images)
    np.random.shuffle(shuffle_ind)
    images = images[shuffle_ind]
    labels = labels[shuffle_ind]
    return images, labels

def one_hot_encode(labels):
    one_hot_labels = np.zeros((len(labels), 10), dtype=np.float32)
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels

def get_train_and_val():
    images_all, labels_all = get_img_and_label('./data/cifar-10-batches-py/data_batch_1')
    for x in range(2, 6):
        file = './data/cifar-10-batches-py/data_batch_' + str(x)
        images, labels = get_img_and_label(file)
        images_all = np.concatenate((images_all, images))
        labels_all = np.concatenate((labels_all, labels))
    images_all, labels_all = simultaneous_shuffle(images_all, labels_all)
    num_images_all = images_all.shape[0]
    num_images_train = int(num_images_all * train_percent)
    images_train = images_all[:num_images_train]
    images_val = images_all[num_images_train:]
    labels_train = labels_all[:num_images_train]
    labels_val = labels_all[num_images_train:]
    return images_train, one_hot_encode(labels_train), images_val, one_hot_encode(labels_val)

def get_test():
    images_test, labels_test = get_img_and_label('./data/cifar-10-batches-py/test_batch')
    num_images_test = images_test.shape[0]
    return images_test, one_hot_encode(labels_test)

def augmentation(images):
    for i in range(images.shape[0]):
        if np.random.rand() < flip_probability:
            images[i] = np.fliplr(images[i])
    return images

class DataLoader:

    def __init__(self):
        self.train_X, self.train_Y, self.val_X, self.val_Y = get_train_and_val()
        self.test_X, self.test_Y = get_test()
        self.train_ind, self.val_ind, self.test_ind = 0, 0, 0
        self.n_train, self.n_val, self.n_test = self.train_X.shape[0], self.val_X.shape[0], self.test_X.shape[0]

    def get_batch(self, batch_size, tag):
        if tag == 'train':
            X, Y = augmentation(self.train_X[self.train_ind : self.train_ind + batch_size]), self.train_Y[self.train_ind : self.train_ind + batch_size]
            self.train_ind += batch_size
        elif tag == 'val':
            X, Y = self.val_X[self.val_ind : self.val_ind + batch_size], self.val_Y[self.val_ind : self.val_ind + batch_size]
            self.val_ind += batch_size
        elif tag == 'test':
            X, Y = self.test_X[self.test_ind : self.test_ind + batch_size], self.test_Y[self.test_ind : self.test_ind + batch_size]
            self.test_ind += batch_size
        return X, Y

    def reset_epoch(self):
        self.train_ind, self.val_ind, self.test_ind = 0, 0, 0
        self.train_X, self.train_Y = simultaneous_shuffle(self.train_X, self.train_Y)

    def num_batches(self, batch_size, tag):
        if tag == 'train':
            n_batches = int(self.n_train / batch_size)
        elif tag == 'val':
            n_batches = int(self.n_val / batch_size)
        elif tag == 'test':
            n_batches = int(self.n_test / batch_size)
        return n_batches

if __name__ == "__main__":
    data_loader = DataLoader()
    # X - [batch_size, height, width, channels]
    # Y - [batch_size, num_classes]
    X = data_loader.num_batches(16, "train")
    print("train", X)
    X = data_loader.num_batches(16, "val")
    print("val", X)
    X = data_loader.num_batches(16, "test")
    print("test", X)
