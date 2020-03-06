# Tensorflow CIFAR-10

This is a Tensorflow implementation of various deep learning models on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Usage
Create a folder 'data' in the repository.

Download and extract [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) in the 'data' folder.

Download [vgg16_weights.npz](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) and place it in the main directory of the repository.
Needed to run the VGG16 model.

To run the various models
```
python train.py feedforward
python train.py CNN
python train.py VGG16
```

## Result
The accuracies on different models is tabulated below

| Model       | Accuracy |
| ----------- | -------: |
| Feedforward | ~51%     |
| CNN         | ~73%     |
| VGG16       | ~84%     |
| ResNet      | TBD      |