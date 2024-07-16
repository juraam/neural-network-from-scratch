import numpy as np
from typing import List, Tuple, Optional

from src.core.network_base import NetworkModule

class ReluLayer(NetworkModule):
  def __init__(self) -> None:
    pass

  def zero_gradient(self):
    pass

  def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    return []

  def forward(self, X: np.ndarray) -> np.ndarray:
    # return value: max(x, 0)
    result = X.copy()
    result[result <= 0] = 0
    self.prev_X = X.copy()

    return result

  def backward(self, loss: np.ndarray) -> np.ndarray:
    result = loss.copy()
    result[self.prev_X <= 0] = 0

    return result