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
python train.py ResNet
```

## Dependencies
```
Python 3 (used 3.6.9)
Tensorflow 1.0 (used 1.7.1)
Albumentations (https://github.com/albumentations-team/albumentations)
```

## Results
The accuracies on different models is tabulated below

| Model       | Accuracy | Initial Learning Rate | Batch Size | Max Epochs |
| ----------- | -------: | --------------------: | ---------: | ---------: |
| Feedforward | ~56%     | 0.0001                | 128        | 30         |
| CNN         | ~77%     | 0.001                 | 128        | 10         |
| VGG16       | ~86%     | 0.0001                | 64         | 10         |
| ResNet-50   | ~86%     | 0.003                 | 32         | 30         |

Note: The VGG16 model was finetuned to the CIFAR-10 dataset after loading pretrained weights trained over the ImageNet dataset while the ResNet-50 model was trained from scratch. This is why the VGG16 model performs as well as the ResNet model.
