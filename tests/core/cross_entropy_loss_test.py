from typing import List, Optional, Tuple
import unittest
import numpy as np

import torch

from src.core.losses import CrossEntropyLoss
from src.core.network_base import NetworkModule

class MockNetworkModule(NetworkModule):
  def zero_gradient(self):
    pass

  def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    return []
  
  def forward(self, X: np.ndarray) -> np.ndarray:
    return X

  def backward(self, loss: np.ndarray) -> np.ndarray:
    return loss

class CrossEntropyTest(unittest.TestCase):
  def test_exception_when_classes_are_hotvectors(self):
    loss = CrossEntropyLoss(MockNetworkModule())
    self.assertRaises(
      Exception,
      loss,
      np.array([1,2]),
      np.array([[0,1], [0,1]])
    )

  def test_exception_when_not_equal_sizes(self):
    loss = CrossEntropyLoss(MockNetworkModule())
    self.assertRaises(
      Exception,
      loss,
      np.array([[1,2,3], [2,3,5]]),
      np.array([0])
    )

  def test_exception_when_classes_max_more_than_size_predicted(self):
    loss = CrossEntropyLoss(MockNetworkModule())
    self.assertRaises(
      Exception,
      loss,
      np.array([[1,2,3], [2,3,5]]),
      np.array([0, 8])
    )

  def test_one_row(self):
    loss = CrossEntropyLoss(MockNetworkModule())
    pred_y = np.array([[1,2,5]])
    target_y = np.array([0])
    output = loss(pred_y, target_y)
    np.testing.assert_almost_equal(
      output,
      4.0659,
      decimal=4
    )
    np.testing.assert_almost_equal(
      loss.grad,
      np.array([[-0.98285217,  0.04661262,  0.93623955]]),
      decimal=8
    )

    torch_loss = torch.nn.CrossEntropyLoss()
    torch_pred_y = torch.tensor(pred_y, dtype=float, requires_grad=True)
    torch_target_y = torch.tensor(target_y, dtype=int)
    torch_output = torch_loss(torch_pred_y, torch_target_y)
    torch_output.backward()

    np.testing.assert_almost_equal(
      torch_output.detach().numpy(),
      output,
      decimal=4
    )
    np.testing.assert_almost_equal(
      loss.grad,
      torch_pred_y.grad.detach().numpy(),
      decimal=8
    )

  def test_batch_sum(self):
    loss = CrossEntropyLoss(MockNetworkModule(), reduction_mode='sum')
    pred_y = np.array([[1,2,5], [2,4,6]])
    target_y = np.array([0,2])
    output = loss(pred_y, target_y)
    np.testing.assert_almost_equal(
      output,
      4.0659 + 0.1429,
      decimal=4
    )
    np.testing.assert_almost_equal(
      loss.grad,
      np.array([
        [-0.98285217,  0.04661262,  0.93623955],
        [0.01587624,  0.11731043, -0.13318667]
      ]),
      decimal=8
    )

    torch_loss = torch.nn.CrossEntropyLoss(reduction='sum')
    torch_pred_y = torch.tensor(pred_y, dtype=float, requires_grad=True)
    torch_target_y = torch.tensor(target_y, dtype=int)
    torch_output = torch_loss(torch_pred_y, torch_target_y)
    torch_output.backward()

    np.testing.assert_almost_equal(
      torch_output.detach().numpy(),
      output,
      decimal=4
    )
    np.testing.assert_almost_equal(
      loss.grad,
      torch_pred_y.grad.detach().numpy(),
      decimal=8
    )

  def test_batch_mean(self):
    loss = CrossEntropyLoss(MockNetworkModule(), reduction_mode='mean')
    pred_y = np.array([[1,2,5], [2,4,6]])
    target_y = np.array([0,2])
    output = loss(pred_y, target_y)
    np.testing.assert_almost_equal(
      output,
      (4.0659 + 0.1429) / 2,
      decimal=4
    )
    np.testing.assert_almost_equal(
      loss.grad,
      np.array([
        [-0.98285217,  0.04661262,  0.93623955],
        [0.01587624,  0.11731043, -0.13318667]
      ]) / 2,
      decimal=8
    )

    torch_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    torch_pred_y = torch.tensor(pred_y, dtype=float, requires_grad=True)
    torch_target_y = torch.tensor(target_y, dtype=int)
    torch_output = torch_loss(torch_pred_y, torch_target_y)
    torch_output.backward()

    np.testing.assert_almost_equal(
      torch_output.detach().numpy(),
      output,
      decimal=4
    )
    np.testing.assert_almost_equal(
      loss.grad,
      torch_pred_y.grad.detach().numpy(),
      decimal=8
    )


if __name__ == "__main__":
  unittest.main()