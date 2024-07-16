import unittest
import numpy as np

import torch

from src.core.fc_layer import FullConnectedLayer

def get_torch_layer(layer: FullConnectedLayer) -> torch.nn.Linear:
  torch_layer = torch.nn.Linear(layer.input_size, layer.output_size)
  torch_layer.weight = torch.nn.Parameter(torch.tensor(layer.weights.transpose(), dtype=float))
  torch_layer.bias = torch.nn.Parameter(torch.tensor(layer.bias, dtype=float))
  return torch_layer

class FullConnectedLayerTest(unittest.TestCase):
  def test_constructor_input_2_output_4(self):
    layer = FullConnectedLayer(input_size=2, output_size=4)
    self.assertEqual(
      layer.weights.shape,
      (2,4)
    )
    self.assertEqual(
      layer.bias.shape,
      (4,)
    )
    self.assertIsNone(layer.bias_grad)
    self.assertIsNone(layer.weight_grad)
  
  def test_constructor_input_5_output_20(self):
    layer = FullConnectedLayer(input_size=5, output_size=20)
    self.assertEqual(
      layer.weights.shape,
      (5,20)
    )
    self.assertEqual(
      layer.bias.shape,
      (20,)
    )
    self.assertIsNone(layer.bias_grad)
    self.assertIsNone(layer.weight_grad)

  def test_raise_exception_when_weights_shape_invalid(self):
    self.assertRaises(Exception, FullConnectedLayer, 2, 4, np.array([1]))
    self.assertRaises(Exception, FullConnectedLayer, 2, 4, None, np.array([1]))

  def test_reset_gradients(self):
    layer = FullConnectedLayer(input_size=5, output_size=20)
    layer.weight_grad = np.array([0])
    layer.bias_grad = np.array([1])
    self.assertIsNotNone(layer.bias_grad)
    self.assertIsNotNone(layer.weight_grad)
    layer.zero_gradient()
    self.assertIsNone(layer.bias_grad)
    self.assertIsNone(layer.weight_grad)

  def test_parameters(self):
    layer = FullConnectedLayer(input_size=2, output_size=2)
    layer.bias = 1
    layer.weights = 2
    layer.weight_grad = 3
    layer.bias_grad = 4
    self.assertEqual(
      layer.parameters(),
      [(1, 4), (2, 3)]
    )

  def test_forward_single_row(self):
    weights = np.array(
      [[1,1], [1,1], [1,1], [1,1], [1,1]]
    )
    bias = np.array([1,1])
    layer = FullConnectedLayer(
      input_size=5,
      output_size=2,
      weights=weights,
      bias=bias
    )

    X = np.array([1,1,1,1,1])
    output = layer.forward(X)

    np.testing.assert_equal(
      layer.prev_X,
      np.expand_dims(X, axis=0)
    )
    np.testing.assert_equal(
      output,
      np.array([[6,6]])
    )

    torch_layer = get_torch_layer(layer)
    torch_x = torch.tensor(X, dtype=float)
    torch_output = torch_layer.forward(torch_x)
    np.testing.assert_equal(
      torch_output.detach().numpy(),
      output.squeeze()
    )
  
  def test_forward_batch(self):
    weights = np.array(
      [[1,2,3,4], [1,2,3,4], [1,2,3,4]]
    )
    bias = np.array([1,2,3,4])
    layer = FullConnectedLayer(
      input_size=3,
      output_size=4,
      weights=weights,
      bias=bias
    )

    X = np.array([[1,1,1], [1,2,3]])
    output = layer.forward(X)

    np.testing.assert_equal(
      layer.prev_X,
      X
    )
    np.testing.assert_equal(
      output,
      np.array([
        [4,8,12,16],
        [7,14,21,28]
      ])
    )

    torch_layer = get_torch_layer(layer)
    torch_x = torch.tensor(X, dtype=float)
    torch_output = torch_layer.forward(torch_x)
    np.testing.assert_equal(
      torch_output.detach().numpy(),
      output
    )

  def test_backward_single(self):
    weights = np.array(
      [[1,1,1], [1,1,1]]
    )
    bias = np.array([1,1,1])
    layer = FullConnectedLayer(
      input_size=2,
      output_size=3,
      weights=weights,
      bias=bias
    )

    X = np.array([1,1])
    layer.forward(X)
    backward_y = np.array([-1, -1, -1])
    output = layer.backward(backward_y)
    np.testing.assert_equal(
      layer.weight_grad,
      np.array([
        [-1, -1, -1],
        [-1, -1, -1],
      ])
    )
    np.testing.assert_equal(
      layer.bias_grad,
      np.array([-1, -1, -1])
    )
    np.testing.assert_equal(
      output,
      np.array([[-3, -3]])
    )

    torch_layer = get_torch_layer(layer)
    torch_x = torch.tensor(X, dtype=float, requires_grad=True)
    torch_backward_y = torch.tensor(backward_y, dtype=float)
    torch_forward = torch_layer.forward(torch_x)
    torch_forward.backward(torch_backward_y)
    np.testing.assert_equal(
      torch_x.grad.detach().numpy(),
      output.squeeze()
    )
    np.testing.assert_equal(
      torch_layer.weight.grad.detach().numpy(),
      layer.weight_grad.transpose()
    )
    np.testing.assert_equal(
      torch_layer.bias.grad.detach().numpy(),
      layer.bias_grad
    )

  def test_backward_batch(self):
    layer = FullConnectedLayer(
      input_size=2,
      output_size=3,
      weights=np.array(
        [[1,2,3], [1,2,3]]
      ),
      bias=np.array([1,2,3])
    )

    X = np.array([[1,2], [3,4]])
    layer.forward(X)
    backward_y = np.array([[-2, 1, 2], [-1,0,1]])
    output = layer.backward(backward_y)
    np.testing.assert_equal(
      layer.weight_grad,
      np.array([
        [-5, 1, 5],
        [-8, 2, 8],
      ])
    )
    np.testing.assert_equal(
      layer.bias_grad,
      np.array([-3, 1, 3])
    )
    np.testing.assert_equal(
      output,
      np.array([
        [6, 6],
        [2, 2]
      ])
    )

    torch_layer = get_torch_layer(layer)
    torch_x = torch.tensor(X, dtype=float, requires_grad=True)
    torch_backward_y = torch.tensor(backward_y, dtype=float)
    torch_forward = torch_layer.forward(torch_x)
    torch_forward.backward(torch_backward_y)
    np.testing.assert_equal(
      torch_x.grad.detach().numpy(),
      output.squeeze()
    )
    np.testing.assert_equal(
      torch_layer.weight.grad.detach().numpy(),
      layer.weight_grad.transpose()
    )
    np.testing.assert_equal(
      torch_layer.bias.grad.detach().numpy(),
      layer.bias_grad
    )

if __name__ == "__main__":
  unittest.main()