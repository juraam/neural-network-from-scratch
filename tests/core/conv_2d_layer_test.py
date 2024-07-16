import unittest
import random

import numpy as np
import torch

from src.core.conv_2d_layer import Convolutional2DLayer

def get_torch_layer(layer: Convolutional2DLayer) -> torch.nn.Conv2d:
  torch_layer = torch.nn.Conv2d(
    in_channels=layer.input_channels,
    out_channels=layer.output_channels,
    kernel_size=layer.kernel_size,
    stride=layer.stride,
    padding=layer.padding
  )
  torch_layer.weight = torch.nn.Parameter(torch.tensor(layer.weights, dtype=float))
  torch_layer.bias = torch.nn.Parameter(torch.tensor(layer.bias, dtype=float))
  return torch_layer

class Convolutional2DLayerTest(unittest.TestCase):
  def test_constructor_input_2_output_4_kernel_3(self):
    layer = Convolutional2DLayer(2, 4, 3)
    self.assertEqual(
      layer.weights.shape,
      (4, 2, 3, 3)
    )
    self.assertEqual(
      layer.bias.shape,
      (4,)
    )
    self.assertIsNone(layer.bias_grad)
    self.assertIsNone(layer.weight_grad)
  
  def test_constructor_input_5_output_20_kernel_10(self):
    layer = Convolutional2DLayer(
      input_channels=5,
      output_channels=20,
      kernel_size=10
    )
    self.assertEqual(
      layer.weights.shape,
      (20,5,10,10)
    )
    self.assertEqual(
      layer.bias.shape,
      (20,)
    )
    self.assertIsNone(layer.bias_grad)
    self.assertIsNone(layer.weight_grad)

  def test_reset_gradients(self):
    layer = Convolutional2DLayer(3,4,5)
    layer.weight_grad = np.array([0])
    layer.bias_grad = np.array([1])
    self.assertIsNotNone(layer.bias_grad)
    self.assertIsNotNone(layer.weight_grad)
    layer.zero_gradient()
    self.assertIsNone(layer.bias_grad)
    self.assertIsNone(layer.weight_grad)

  def test_parameters(self):
    layer = Convolutional2DLayer(3,4,5)
    layer.bias = 1
    layer.weights = 2
    layer.weight_grad = 3
    layer.bias_grad = 4
    self.assertEqual(
      layer.parameters(),
      [(1, 4), (2, 3)]
    )

  def test_raise_forward_without_batches(self):
    layer = Convolutional2DLayer(3, 4, 5)
    self.assertRaises(Exception, layer.forward, np.zeros((3, 5, 5)))
  
  def test_raise_forward_with_invlaid_input_channel(self):
    layer = Convolutional2DLayer(3, 4, 5)
    self.assertRaises(Exception, layer.forward, np.zeros((2, 2, 5, 5)))

  def test_raise_forward_with_invlaid_new_width(self):
    layer = Convolutional2DLayer(3, 4, 5, stride=2)
    self.assertRaises(Exception, layer.forward, np.zeros((2, 3, 5, 6)))

  def test_raise_forward_with_invlaid_new_height(self):
    layer = Convolutional2DLayer(3, 4, 5, stride=2)
    self.assertRaises(Exception, layer.forward, np.zeros((2, 3, 6, 5)))

  def test_forward_okay_with_stride_and_padding(self):
    layer = Convolutional2DLayer(3, 4, 5, stride=2, padding=1)
    layer.forward(np.zeros((2, 3, 7, 7)))

  def test_forward_raise_with_padding_not_divide_by_stride(self):
    layer = Convolutional2DLayer(3, 4, 5, stride=3, padding=1)
    self.assertRaises(Exception, layer.forward, np.zeros((2, 3, 7, 7)))

  def test_forward_single_row(self):
    kernel = np.array([
      [1,2],
      [3,4]
    ])
    weights = np.broadcast_to(kernel, (2, 2,) + kernel.shape)
    self.assertEqual(
      weights.shape,
      (2, 2, 2, 2)
    )
    bias = np.array([1,2])
    layer = Convolutional2DLayer(
      input_channels=2,
      output_channels=2,
      kernel_size=2
    )
    layer.bias = bias
    layer.weights = weights

    X = np.array([[
      [
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 0]
      ],
      [
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0]
      ]
    ]])
    output = layer.forward(X)

    np.testing.assert_equal(
      layer.prev_X,
      X
    )
    np.testing.assert_equal(
      output,
      np.array([[
        [
          [12, 16],
          [16, 12]
        ],
        [
          [13, 17],
          [17, 13]
        ]
      ]])
    )

    torch_layer = get_torch_layer(layer)
    torch_x = torch.tensor(X, dtype=float)
    torch_output = torch_layer.forward(torch_x)
    np.testing.assert_equal(
      torch_output.detach().numpy(),
      output
    )
  
  def test_forward_batch(self):
    weights = np.random.random((5, 3, 3, 3))
    bias = np.random.random(5)
    layer = Convolutional2DLayer(
      input_channels=3,
      output_channels=5,
      kernel_size=3,
      stride=2,
      padding=2
    )
    layer.bias = bias
    layer.weights = weights

    X = np.random.random((6, 3, 5, 5))
    output = layer.forward(X)

    torch_layer = get_torch_layer(layer)
    torch_x = torch.tensor(X, dtype=float)
    torch_output = torch_layer.forward(torch_x)
    np.testing.assert_almost_equal(
      torch_output.detach().numpy(),
      output,
      decimal=5
    )

  def test_backward_single_row(self):
    kernel = np.array([
      [1,2],
      [3,4]
    ])
    weights = np.broadcast_to(kernel, (2, 2,) + kernel.shape)
    bias = np.array([1,2])
    layer = Convolutional2DLayer(
      input_channels=2,
      output_channels=2,
      kernel_size=2
    )
    layer.bias = bias
    layer.weights = weights

    X = np.array([[
      [
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 0]
      ],
      [
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0]
      ]
    ]])
    layer.forward(X)
    backward_y = np.array([[
      [
        [1, -1],
        [0, 2]
      ],
      [
        [-1, 0],
        [1, 1]
      ]
    ]])
    output = layer.backward(backward_y)

    np.testing.assert_equal(
      layer.weight_grad,
      np.array([
        [
          [
            [3, 1],
            [2, 0]
          ],
          [
            [0, 1],
            [3, -1]
          ]
        ],
        [
          [
            [1, 2],
            [0, 0]
          ],
          [
            [1, 1],
            [1, 1]
          ]
        ]
      ])
    )

    np.testing.assert_equal(
      output,
      np.array([[
        [
          [0, -1, -2],
          [1, 2, 2],
          [3, 13, 12]
        ],
        [
          [0, -1, -2],
          [1, 2, 2],
          [3, 13, 12]
        ]
      ]])
    )

    np.testing.assert_equal(
      layer.bias_grad,
      np.array([2, 1])
    )

    torch_layer = get_torch_layer(layer)
    torch_x = torch.tensor(X, dtype=float, requires_grad=True)
    torch_forward = torch_layer.forward(torch_x)
    torch_backward_y = torch.tensor(backward_y, dtype=float)
    torch_forward.backward(torch_backward_y)

    np.testing.assert_equal(
      torch_layer.weight.grad.detach().numpy(),
      layer.weight_grad
    )
    np.testing.assert_equal(
      torch_layer.bias.grad.detach().numpy(),
      layer.bias_grad
    )
    np.testing.assert_equal(
      torch_x.grad.detach().numpy(),
      output
    )

  def test_backward_batch(self):
    weights = np.random.random((5, 3, 3, 3))
    bias = np.random.random(5)
    layer = Convolutional2DLayer(
      input_channels=3,
      output_channels=5,
      kernel_size=3,
      stride=2,
      padding=2
    )
    layer.bias = bias
    layer.weights = weights

    X = np.random.random((6, 3, 5, 5))
    backward_y = np.random.random((6, 5, 4, 4))
    layer.forward(X)
    output = layer.backward(backward_y)

    torch_layer = get_torch_layer(layer)
    torch_x = torch.tensor(X, dtype=float, requires_grad=True)
    torch_forward = torch_layer.forward(torch_x)
    torch_backward_y = torch.tensor(backward_y, dtype=float)
    torch_forward.backward(torch_backward_y)

    np.testing.assert_almost_equal(
      torch_layer.weight.grad.detach().numpy(),
      layer.weight_grad,
      decimal=8
    )
    np.testing.assert_almost_equal(
      torch_layer.bias.grad.detach().numpy(),
      layer.bias_grad,
      decimal=8
    )
    np.testing.assert_almost_equal(
      torch_x.grad.detach().numpy(),
      output,
      decimal=8
    )

  def test_random(self):
    for _ in range(5):
      output_channels = random.randint(1,10)
      input_channels = random.randint(1,10)
      kernel_size = random.randint(2,6)
      stride = random.randint(1,6)
      output_width = random.randint(1,4)
      output_height = random.randint(1,4)
      padding = random.randint(
        0,
        min(
          4,
          ((output_width - 1) * stride + kernel_size) // 2 - 1,
          ((output_height - 1) * stride + kernel_size) // 2 - 1,
        )
      )
      batch_size = random.randint(1,8)
      input_width = (output_width - 1) * stride + kernel_size - 2 * padding
      input_height = (output_height - 1) * stride + kernel_size - 2 * padding
      weights = np.random.random((output_channels, input_channels, kernel_size, kernel_size))
      bias = np.random.random(output_channels)
      layer = Convolutional2DLayer(
        input_channels=input_channels,
        output_channels=output_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
      )
      layer.bias = bias
      layer.weights = weights

      X = np.random.random((batch_size, input_channels, input_width, input_height))
      backward_y = np.random.random((batch_size, output_channels, output_width, output_height))
      layer.forward(X)
      output = layer.backward(backward_y)

      torch_layer = get_torch_layer(layer)
      torch_x = torch.tensor(X, dtype=float, requires_grad=True)
      torch_forward = torch_layer.forward(torch_x)
      torch_backward_y = torch.tensor(backward_y, dtype=float)
      torch_forward.backward(torch_backward_y)

      np.testing.assert_almost_equal(
        torch_layer.weight.grad.detach().numpy(),
        layer.weight_grad,
        decimal=8
      )
      np.testing.assert_almost_equal(
        torch_layer.bias.grad.detach().numpy(),
        layer.bias_grad,
        decimal=8
      )
      np.testing.assert_almost_equal(
        torch_x.grad.detach().numpy(),
        output,
        decimal=8
      )

if __name__ == "__main__":
  unittest.main()