import numpy as np

from src.core.network_base import OptimizerModule, NetworkModule

class SGD(OptimizerModule):
  def __init__(
      self,
      model: NetworkModule,
      lr = 0.0001,
      weight_decay = 0,
      momentum = 0
    ) -> None:
    super().__init__(model)
    self.lr = lr
    self.weight_decay = weight_decay
    self.momentum = momentum
    self.velocities = {}

  def step(self):
    for index, (value, grad) in enumerate(self.model.parameters()):
      if grad is None and value is None:
        continue
      if grad is None:
        raise Exception("Should not call optimize with zero grad")
      if grad.shape != value.shape:
        raise Exception("Should be the same shapes")
      new_grad = grad.copy()
      if self.weight_decay != 0:
        new_grad += self.weight_decay * value
      if self.momentum != 0:
        if index not in self.velocities:
          self.velocities[index] = np.zeros_like(value)
        self.velocities[index] = self.momentum * self.velocities[index] + self.lr * new_grad
        value -= self.velocities[index]
      else:
        value += -(self.lr * new_grad)