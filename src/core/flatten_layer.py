import numpy as np
from typing import List, Tuple, Optional

from src.core.network_base import NetworkModule

class FlattenLayer(NetworkModule):
  prev_shape: Optional[np.ndarray]

  def __init__(self) -> None:
    pass

  def zero_gradient(self):
    pass

  def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    return []

  def forward(self, X: np.ndarray) -> np.ndarray:
    shape = X.shape
    self.prev_shape = shape
    size = 1
    for i in range(1, len(shape)):
      size *= shape[i]
    return X.reshape((shape[0], size))

  def backward(self, loss: np.ndarray) -> np.ndarray:
    shape = self.prev_shape

    return loss.reshape(shape)