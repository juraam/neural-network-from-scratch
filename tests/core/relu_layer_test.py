import unittest
import numpy as np

import torch

from src.core.relu_layer import ReluLayer

class ReluLayerTest(unittest.TestCase):
  def test_single_row(self):
    layer = ReluLayer()

    X = np.array([-2,2,-3,0,8])
    backward_y = np.array([100, 2, -188, 2, 1])
    output = layer.forward(X)
    output_backward = layer.backward(backward_y)

    np.testing.assert_equal(
      layer.prev_X,
      X
    )
    np.testing.assert_equal(
      output,
      np.array([0,2,0,0,8])
    )
    np.testing.assert_equal(
      output_backward,
      np.array([0,2,0,0,1])
    )

    torch_layer = torch.nn.ReLU()
    torch_x = torch.tensor(X, dtype=float, requires_grad=True)
    torch_output = torch_layer.forward(torch_x)
    torch_backward_y = torch.tensor(backward_y, dtype=float)
    torch_forward = torch_layer.forward(torch_x)
    torch_forward.backward(torch_backward_y)
    np.testing.assert_equal(
      torch_x.grad.detach().numpy(),
      output_backward
    )
    np.testing.assert_equal(
      torch_output.detach().numpy(),
      output
    )
  
  def test_batch(self):
    layer = ReluLayer()

    X = np.array([[-2,2,-3,0,8], [12, 22, 1, -2111, -3]])
    backward_y = np.array([[100, 2, -188, 2, 1], [8, -2, 3, 29, 1000]])
    output = layer.forward(X)
    output_backward = layer.backward(backward_y)

    np.testing.assert_equal(
      layer.prev_X,
      X
    )
    np.testing.assert_equal(
      output,
      np.array([[0,2,0,0,8], [12, 22, 1, 0, 0]])
    )
    np.testing.assert_equal(
      output_backward,
      np.array([[0,2,0,0,1], [8, -2, 3, 0, 0]])
    )

    torch_layer = torch.nn.ReLU()
    torch_x = torch.tensor(X, dtype=float, requires_grad=True)
    torch_output = torch_layer.forward(torch_x)
    torch_backward_y = torch.tensor(backward_y, dtype=float)
    torch_forward = torch_layer.forward(torch_x)
    torch_forward.backward(torch_backward_y)
    np.testing.assert_equal(
      torch_x.grad.detach().numpy(),
      output_backward
    )
    np.testing.assert_equal(
      torch_output.detach().numpy(),
      output
    )

  def test_random_batch(self):
    layer = ReluLayer()

    X = np.random.random((10, 4, 20, 20))
    backward_y = np.random.random((10, 4, 20, 20))
    output = layer.forward(X)
    output_backward = layer.backward(backward_y)

    torch_layer = torch.nn.ReLU()
    torch_x = torch.tensor(X, dtype=float, requires_grad=True)
    torch_output = torch_layer.forward(torch_x)
    torch_backward_y = torch.tensor(backward_y, dtype=float)
    torch_forward = torch_layer.forward(torch_x)
    torch_forward.backward(torch_backward_y)
    np.testing.assert_equal(
      torch_x.grad.detach().numpy(),
      output_backward
    )
    np.testing.assert_equal(
      torch_output.detach().numpy(),
      output
    )

if __name__ == "__main__":
  unittest.main()