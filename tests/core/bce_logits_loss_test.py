from typing import List, Optional, Tuple
import unittest
import numpy as np

import torch

from src.core.losses import BCEWithLogitsLoss
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

class BCELogitsTest(unittest.TestCase):
  def test_exception_when_classes_are_scalar(self):
    loss = BCEWithLogitsLoss(MockNetworkModule())
    self.assertRaises(
      Exception,
      loss,
      np.array([4,1]),
      np.array([[0,1], [0,1]])
    )

  def test_one_row_mean(self):
    loss = BCEWithLogitsLoss(MockNetworkModule())
    pred_y = np.array([1,2,5])
    target_y = np.array([0,1,1])
    output = loss(pred_y, target_y)
    np.testing.assert_almost_equal(
      output,
      0.4823,
      decimal=4
    )
    np.testing.assert_almost_equal(
      loss.grad,
      np.array([[0.24368619, -0.03973431, -0.00223095]]),
      decimal=8
    )

    torch_loss = torch.nn.BCEWithLogitsLoss()
    torch_pred_y = torch.tensor(pred_y, dtype=float, requires_grad=True)
    torch_target_y = torch.tensor(target_y, dtype=float)
    torch_output = torch_loss(torch_pred_y, torch_target_y)
    torch_output.backward()

    np.testing.assert_almost_equal(
      torch_output.detach().numpy(),
      output,
      decimal=8
    )
    np.testing.assert_almost_equal(
      loss.grad.squeeze(),
      torch_pred_y.grad.detach().numpy(),
      decimal=8
    )

  def test_one_row_sum(self):
    loss = BCEWithLogitsLoss(MockNetworkModule(), reduction_mode='sum')
    pred_y = np.array([1,2,5])
    target_y = np.array([0,1,1])
    output = loss(pred_y, target_y)
    np.testing.assert_almost_equal(
      output,
      0.4823 * 3,
      decimal=4
    )
    np.testing.assert_almost_equal(
      loss.grad,
      np.array([[0.24368619, -0.03973431, -0.00223095]]) * 3,
      decimal=8
    )

    torch_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
    torch_pred_y = torch.tensor(pred_y, dtype=float, requires_grad=True)
    torch_target_y = torch.tensor(target_y, dtype=float)
    torch_output = torch_loss(torch_pred_y, torch_target_y)
    torch_output.backward()

    np.testing.assert_almost_equal(
      torch_output.detach().numpy(),
      output,
      decimal=8
    )
    np.testing.assert_almost_equal(
      loss.grad.squeeze(),
      torch_pred_y.grad.detach().numpy(),
      decimal=8
    )

  def test_batch_mean(self):
    loss = BCEWithLogitsLoss(MockNetworkModule())
    pred_y = np.array([[1,2,5], [2,4,6]])
    target_y = np.array([[0,1,1], [1,0,1]])
    output = loss(pred_y, target_y)
    np.testing.assert_almost_equal(
      output,
      5.5945 / 6,
      decimal=4
    )
    np.testing.assert_almost_equal(
      loss.grad,
      np.array([
        [0.24368619, -0.03973431, -0.00223095],
        [-0.03973431,  0.32733793, -0.00082421]
      ]) / 2,
      decimal=8
    )

    torch_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    torch_pred_y = torch.tensor(pred_y, dtype=float, requires_grad=True)
    torch_target_y = torch.tensor(target_y, dtype=float)
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
    loss = BCEWithLogitsLoss(MockNetworkModule(), reduction_mode='sum')
    pred_y = np.array([[1,2,5], [2,4,6]])
    target_y = np.array([[0,1,1], [1,0,1]])
    output = loss(pred_y, target_y)
    np.testing.assert_almost_equal(
      output,
      5.5945,
      decimal=4
    )
    np.testing.assert_almost_equal(
      loss.grad,
      np.array([
        [0.24368619, -0.03973431, -0.00223095],
        [-0.03973431,  0.32733793, -0.00082421]
      ]) * 3,
      decimal=8
    )

    torch_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
    torch_pred_y = torch.tensor(pred_y, dtype=float, requires_grad=True)
    torch_target_y = torch.tensor(target_y, dtype=float)
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