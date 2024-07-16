import numpy as np
from typing import List, Tuple, Optional

from src.core.network_base import NetworkModule, translate_to_minibatch

class Convolutional2DLayer(NetworkModule):
  def __init__(
      self,
      input_channels: int,
      output_channels: int,
      kernel_size: int,
      stride: int = 1,
      padding: int = 0
    ) -> None:
    self.input_channels = input_channels
    self.output_channels = output_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.weights = np.random.random((output_channels, input_channels, kernel_size, kernel_size))
    self.bias = np.random.random((output_channels))
    self.bias_grad = None
    self.weight_grad = None

  def zero_gradient(self):
    self.weight_grad = None
    self.bias_grad = None

  def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    return [(self.bias, self.bias_grad), (self.weights, self.weight_grad)]

  def forward(self, X: np.ndarray) -> np.ndarray:
    if len(X.shape) != 4:
      raise Exception("shape should be size of N batches, input_channels, width, height")
    if X.shape[1] != self.input_channels:
      raise Exception(f"input channels of input image {X.shape[1]} not equal to {self.input_channels}")
    N, input_channels, w, h = X.shape
    if (w + 2 * self.padding - (self.kernel_size - 1) - 1) % self.stride != 0:
      raise Exception(f"new width is not integer")
    if (h + 2 * self.padding - (self.kernel_size - 1) - 1) % self.stride != 0:
      raise Exception(f"new height is not integer")
    X = X.copy()
    X = np.pad(
      X,
      (
        (0, 0),
        (0, 0),
        (self.padding, self.padding),
        (self.padding, self.padding)
      )
    )
    self.prev_X = X
    new_width = (w + 2 * self.padding - (self.kernel_size - 1) - 1) // self.stride + 1
    new_height = (h + 2 * self.padding - (self.kernel_size - 1) - 1) // self.stride + 1
    output = np.zeros((
      N,
      self.output_channels,
      new_width,
      new_height
    ))

    for i in range(new_width):
      for j in range(new_height):
        output[:, :, i, j] = np.sum (
          X[
            :,
            np.newaxis,
            :,
            i * self.stride : i * self.stride + self.kernel_size,
            j * self.stride : j * self.stride + self.kernel_size,
          ] * self.weights,
          axis=(2,3,4)
        )
    output += self.bias[np.newaxis, :, np.newaxis, np.newaxis]
    return output

  def backward(self, loss: np.ndarray) -> np.ndarray:
    self.weight_grad = np.zeros_like(self.weights)
    self.bias_grad = np.zeros_like(self.bias)

    N, out_channels, width, height = loss.shape
    dilation_width = width + (self.stride - 1) * (width - 1)
    dilation_height = height + (self.stride - 1) * (height - 1)
    dilation_loss = np.zeros((N, out_channels, dilation_width, dilation_height))
    dilation_loss[:, :, : : self.stride, : : self.stride] = loss

    for i in range(self.kernel_size):
      for j in range(self.kernel_size):
        self.weight_grad[:, :, i, j] = np.sum (
          self.prev_X[
            :,
            np.newaxis,
            :,
            i : i + dilation_width,
            j : j + dilation_height,
          ] * dilation_loss[:, :, np.newaxis, : , :],
          axis=(0,3,4)
        )

    rotated_weights = np.rot90(self.weights, 2, axes=(2,3)).transpose((1, 0, 2, 3))
    dilation_loss = np.pad(
      dilation_loss,
      (
        (0, 0),
        (0, 0),
        (self.kernel_size - 1, self.kernel_size - 1),
        (self.kernel_size - 1, self.kernel_size - 1)
      )
    )

    _, _, old_width, old_height = self.prev_X.shape
    output = np.zeros_like(self.prev_X)
    for i in range(old_width):
      for j in range(old_height):
        output[:, :, i, j] = np.sum (
          dilation_loss[
            :,
            np.newaxis,
            :,
            i : i + self.kernel_size,
            j : j + self.kernel_size,
          ] * rotated_weights,
          axis=(2,3,4)
        )

    if self.padding != 0:
      output = output[:, :, self.padding : -self.padding, self.padding : -self.padding]
    self.bias_grad = np.sum(loss, axis=(0,2,3))
    return output

