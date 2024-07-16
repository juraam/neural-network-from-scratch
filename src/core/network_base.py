from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
import numpy as np

def translate_to_minibatch(x: np.ndarray) -> np.ndarray:
  if len(x.shape) == 1:
    return np.expand_dims(x, axis=0)
  return x

def softmax(x: np.ndarray) -> np.ndarray:
  x = translate_to_minibatch(x)
  exp_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
  return exp_x / np.sum(exp_x, axis=1)[:, np.newaxis]

def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(-x))

class NetworkModule(ABC):
  @abstractmethod
  def zero_gradient(self):
    pass

  @abstractmethod
  def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    pass
  
  @abstractmethod
  def forward(self, X: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  def backward(self, loss: np.ndarray) -> np.ndarray:
    pass

class LossModule(ABC):
  model: NetworkModule
  grad: Optional[np.ndarray]

  def __init__(self, model: NetworkModule) -> None:
    self.model = model

  @abstractmethod
  def __call__(self, predicted_values: np.ndarray, truth_values: np.ndarray) -> float:
    pass

  @abstractmethod
  def backward(self) -> None:
    pass

class OptimizerModule(ABC):
  def __init__(self, model: NetworkModule) -> None:
    self.model = model

  def zero_gradient(self):
    self.model.zero_gradient()

  @abstractmethod
  def step(self):
    pass
  
class LayersNeuralNetwork(NetworkModule):
  def __init__(self, layers: List[NetworkModule]) -> None:
    self.layers = layers

  def zero_gradient(self):
    for layer in self.layers:
      layer.zero_gradient()

  def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    result = []
    for layer in self.layers:
      result.extend(layer.parameters())
    return result
  
  def forward(self, X: np.ndarray) -> np.ndarray:
    output = X
    for layer in self.layers:
      output = layer.forward(output)
    return output
  
  def backward(self, loss: np.ndarray) -> np.ndarray:
    output = loss
    for layer in reversed(self.layers):
      output = layer.backward(output)
    return output
  


  