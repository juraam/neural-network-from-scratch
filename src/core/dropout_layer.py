import numpy as np
from typing import List, Tuple, Optional

from src.core.network_base import NetworkModule

class DropoutLayer(NetworkModule):
  def __init__(self, p: float = 0.5) -> None:
    if p >= 1 or p <= 0:
      raise Exception("probability should me less than 1 and more than 0")
    self.p = p

  def zero_gradient(self):
    pass

  def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    return []

  def forward(self, X: np.ndarray) -> np.ndarray:
    self.mask = np.random.uniform(low=0, high=1, size=X.shape) > self.p
    result = X * self.mask / (1 - self.p)

    return result

  def backward(self, loss: np.ndarray) -> np.ndarray:
    result = loss * self.mask / (1 - self.p)

    return result