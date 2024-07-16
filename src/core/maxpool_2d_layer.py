from typing import List, Tuple, Optional
import math

import numpy as np

from src.core.network_base import NetworkModule

class MaxPool2DLayer(NetworkModule):
  def __init__(
      self,
      kernel_size: int,
      stride: Optional[int] = None,
      padding: int = 0
    ) -> None:
    self.kernel_size = kernel_size
    self.stride = kernel_size if stride is None else stride
    self.padding = padding

  def zero_gradient(self):
    pass

  def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    return []

  def forward(self, X: np.ndarray) -> np.ndarray:
    if len(X.shape) != 4:
      raise Exception("shape should be size of N batches, channels, width, height")
    N, channels, w, h = X.shape
    new_width = math.floor((w + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
    new_height = math.floor((h + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
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
    output = np.zeros((
      N,
      channels,
      new_width,
      new_height
    ))

    self.prev_X = X
    for i in range(new_width):
      for j in range(new_height):
        pool_region = X[
          :,
          :,
          i * self.stride : i * self.stride + self.kernel_size,
          j * self.stride : j * self.stride + self.kernel_size,
        ]
        pooled_value = np.max(pool_region, axis=(2, 3))
        output[:, :, i, j] = pooled_value
    return output

  def backward(self, loss: np.ndarray) -> np.ndarray:
    output = np.zeros_like(self.prev_X)

    _, _, width, height = loss.shape

    for i in range(width):
      for j in range(height):
        pool_region = self.prev_X[
          :,
          :,
          i * self.stride : i * self.stride + self.kernel_size,
          j * self.stride : j * self.stride + self.kernel_size,
        ]
        pool_mask = pool_region == np.max(pool_region, axis=(2, 3), keepdims=True)
        output[
          :,
          :,
          i * self.stride : i * self.stride + self.kernel_size,
          j * self.stride : j * self.stride + self.kernel_size,
        ] += loss[:, :, i, j][:, :, np.newaxis, np.newaxis] * pool_mask

    if self.padding != 0:
      output = output[:, :, self.padding : -self.padding, self.padding : -self.padding]
    return output

