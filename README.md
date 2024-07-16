# Neural Network from scratch with numpy

## Little intro

It is an educational repository with a purpose to understand how base layers (such as linear, maxpool, conv2d) work.

## Implemented Layers

1. Full Connected Layer (Linear)
2. Dropout Layer
3. Convolutional 2d Layer
4. Flatten Layer
5. ReLU Layer
6. MaxPool2d Layer

## Implemented Optimizers

* SGD

## Implemented Losses

1. Cross Entropy Loss
2. BCEWithLogitsLoss

## Code

### INSTALL REQUIREMENTS

To start working with code, please download all required dependencies:

```shell
pip install -r requirements.txt
```

### MNIST SOLUTION

For educational purpose I made a [Simple Solution](https://github.com/juraam/stable-diffusion-from-scratch/blob/main/mnist_solution.ipynb) to regognize MNIST digits.

### RUN TESTS

All layers are covered by tests. In all tests I compare results from my layers and torch implementation. And with the same input data, same parameters of a layers they should give the same results.

To run tests, call this command in a terminal:
```shell
python -m unittest discover -v -s ./tests/core -p "*_test.py"
```
