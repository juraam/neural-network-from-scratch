from typing import List, Optional, Tuple
import unittest
import numpy as np

import torch

from src.core.optimizers import SGD
from src.core.network_base import NetworkModule

class MockNetworkModule(NetworkModule):
  def __init__(self, values: np.ndarray, grads: Optional[np.ndarray]) -> None:
    self.values = values
    self.grads = grads

  def zero_gradient(self):
    pass

  def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    return [(self.values, self.grads)]
  
  def forward(self, X: np.ndarray) -> np.ndarray:
    return X

  def backward(self, loss: np.ndarray) -> np.ndarray:
    return loss

class SGDTest(unittest.TestCase):
  def test_exception_when_grad_none(self):
    model = MockNetworkModule(
      values=np.array([[2,3,4], [5,6,7]]),
      grads=None
    )
    optimizer = SGD(
      model=model,
      lr=1
    )
    self.assertRaises(Exception, optimizer.step)

  def test_exception_when_different_shapes(self):
    model = MockNetworkModule(
      values=np.array([[2,3,4], [5,6,7]]),
      grads=np.array([[2,3,4]])
    )
    optimizer = SGD(
      model=model,
      lr=1
    )
    self.assertRaises(Exception, optimizer.step)

  def test_change_params_with_lr_1(self):
    model = MockNetworkModule(
      values=np.array([[2,3,4], [5,6,7]]),
      grads=np.array([[1,5,10], [2,3,4]])
    )
    torch_values = torch.tensor(model.values, dtype=float)
    torch_values.grad = torch.tensor(model.grads, dtype=float)
    optimizer = SGD(
      model=model,
      lr=1
    )
    optimizer.step()

    np.testing.assert_equal(
      model.values,
      np.array([
        [1,-2,-6],
        [3,3,3]
      ])
    )

    torch_optimizer = torch.optim.SGD([torch_values], lr=1)
    torch_optimizer.step()
    np.testing.assert_equal(
      model.values,
      torch_values.detach().numpy()
    )

  def test_change_params_with_lr_3(self):
    model = MockNetworkModule(
      values=np.array([[2,3,4], [5,6,7]]),
      grads=np.array([[1,5,10], [2,3,4]])
    )
    torch_values = torch.tensor(model.values, dtype=float)
    torch_values.grad = torch.tensor(model.grads, dtype=float)

    optimizer = SGD(
      model=model,
      lr=3
    )
    optimizer.step()

    np.testing.assert_equal(
      model.values,
      np.array([
        [-1,-12,-26],
        [-1,-3,-5]
      ])
    )

    torch_optimizer = torch.optim.SGD([torch_values], lr=3)
    torch_optimizer.step()
    np.testing.assert_equal(
      model.values,
      torch_values.detach().numpy()
    )

  def test_change_params_with_decay_1(self):
    model = MockNetworkModule(
      values=np.array([[2,3,4]]),
      grads=np.array([[1,5,10]])
    )
    torch_values = torch.tensor(model.values, dtype=float)
    torch_values.grad = torch.tensor(model.grads, dtype=float)

    optimizer = SGD(
      model=model,
      lr=1,
      weight_decay=1
    )
    optimizer.step()

    np.testing.assert_equal(
      model.values,
      np.array([
        [-1,-5,-10]
      ])
    )

    torch_optimizer = torch.optim.SGD([torch_values], lr=1, weight_decay=1)
    torch_optimizer.step()
    np.testing.assert_equal(
      model.values,
      torch_values.detach().numpy()
    )

  def test_change_params_with_momentum_1(self):
    model = MockNetworkModule(
      values=np.array([[2,3,4]]),
      grads=np.array([[1,5,10]])
    )
    torch_values = torch.tensor(model.values, dtype=float)
    torch_values.grad = torch.tensor(model.grads, dtype=float)

    optimizer = SGD(
      model=model,
      lr=1,
      weight_decay=1,
      momentum=1
    )
    optimizer.step()
    optimizer.step()
    optimizer.step()

    torch_optimizer = torch.optim.SGD(
      [torch_values],
      lr=1,
      weight_decay=1,
      momentum=1
    )
    torch_optimizer.step()
    torch_optimizer.step()
    torch_optimizer.step()
    np.testing.assert_equal(
      model.values,
      torch_values.detach().numpy()
    )

if __name__ == "__main__":
  unittest.main()