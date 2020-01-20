# Tensorflow CIFAR-10

This is a Tensorflow implementation of various deep learning models on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Usage
Create a folder 'data' in the repository.

Download and extract [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) in the 'data' folder.

To run the various models
```
python train.py feedforward
python train.py CNN
```

## Result
The accuracies on different models is tabulated below

| Model       | Accuracy |
| ----------- | -------: |
| Feedforward | ~51%     |
| CNN         | ~73%     |