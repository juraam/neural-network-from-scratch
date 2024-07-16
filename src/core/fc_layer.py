import numpy as np
from typing import List, Tuple, Optional

from src.core.network_base import NetworkModule, translate_to_minibatch

class FullConnectedLayer(NetworkModule):
  def __init__(
      self,
      input_size: int,
      output_size: int,
      weights: Optional[np.ndarray] = None,
      bias: Optional[np.ndarray] = None
    ) -> None:
    # W = (input x output)
    # Bias(b) = (output x 1)
    self.input_size = input_size
    self.output_size = output_size
    if weights is None:
      self.weights = np.random.random((input_size, output_size))
    else:
      if weights.shape != (input_size, output_size):
        raise Exception("not equals shape for weights")
      self.weights = weights
    if bias is None:
      self.bias = np.random.random((output_size))
    else:
      if bias.shape != (output_size, ):
        raise Exception("not equals shape for bias")
      self.bias = bias
    self.bias_grad = None
    self.weight_grad = None

  def zero_gradient(self):
    self.bias_grad = None
    self.weight_grad = None

  def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    return [(self.bias, self.bias_grad), (self.weights, self.weight_grad)]

  def forward(self, X: np.ndarray) -> np.ndarray:
    X = translate_to_minibatch(X)
    if len(X.shape) != 2:
      raise Exception("No batches")
    self.prev_X = X.copy()
    # Forward propagation: X * W + b
    return X.dot(self.weights) + np.tile(self.bias, ((X.shape[0], 1)))

  def backward(self, loss: np.ndarray) -> np.ndarray:
    loss = translate_to_minibatch(loss)
    # Backward propagation
    # d_W = loss * X
    # d_b = loss
    self.bias_grad = loss.sum(axis=0)
    self.weight_grad = self.prev_X.transpose().dot(loss)

    # return value: loss * W
    return loss.dot(self.weights.transpose())