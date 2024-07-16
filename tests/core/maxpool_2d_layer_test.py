import unittest
import random

import numpy as np
import torch

from src.core.maxpool_2d_layer import MaxPool2DLayer

def get_torch_layer(layer: MaxPool2DLayer) -> torch.nn.MaxPool2d:
  torch_layer = torch.nn.MaxPool2d(
    kernel_size=layer.kernel_size,
    stride=layer.stride,
    padding=layer.padding
  )
  return torch_layer

class MaxPool2DLayerTest(unittest.TestCase):
  def test_raise_forward_without_batches(self):
    layer = MaxPool2DLayer(kernel_size=2)
    self.assertRaises(Exception, layer.forward, np.zeros((3, 5, 5)))

  def test_forward_and_backward_single_row(self):
    layer = MaxPool2DLayer(
      kernel_size=2,
      stride=1
    )

    X = np.array([[
      [
        [1, 0, 2],
        [3, 4, 5],
        [0, 6, 0]
      ],
      [
        [0, 0, -1],
        [1, 0, 2],
        [4, 3, 0]
      ]
    ]])
    output = layer.forward(X)

    np.testing.assert_equal(
      output,
      np.array([[
        [
          [4, 5],
          [6, 6]
        ],
        [
          [1, 2],
          [4, 3]
        ]
      ]])
    )

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
    backward_output = layer.backward(backward_y)
    np.testing.assert_equal(
      backward_output,
      np.array([[
        [
          [0, 0, 0],
          [0, 1, -1],
          [0, 2, 0]
        ],
        [
          [0, 0, 0],
          [-1, 0, 0],
          [1, 1, 0]
        ]
      ]])
    )

    torch_layer = get_torch_layer(layer)
    torch_x = torch.tensor(X, dtype=float, requires_grad=True)
    torch_forward = torch_layer.forward(torch_x)
    torch_backward_y = torch.tensor(backward_y, dtype=float)
    torch_forward.backward(torch_backward_y)
    np.testing.assert_equal(
      torch_forward.detach().numpy(),
      output
    )
    np.testing.assert_almost_equal(
      torch_x.grad.detach().numpy(),
      backward_output,
      decimal=8
    )

  def test_forward_and_backward_single_row_witn_one_max(self):
    layer = MaxPool2DLayer(
      kernel_size=2,
      stride=1
    )

    X = np.array([[
      [
        [1, 0, 2],
        [3, 4, 3],
        [0, 1, 0]
      ]
    ]])
    output = layer.forward(X)

    np.testing.assert_equal(
      output,
      np.array([[
        [
          [4, 4],
          [4, 4]
        ]
      ]])
    )

    backward_y = np.array([[
      [
        [1, -1],
        [0, 2]
      ]
    ]])
    backward_output = layer.backward(backward_y)
    np.testing.assert_equal(
      backward_output,
      np.array([[
        [
          [0, 0, 0],
          [0, 2, 0],
          [0, 0, 0]
        ]
      ]])
    )

    torch_layer = get_torch_layer(layer)
    torch_x = torch.tensor(X, dtype=float, requires_grad=True)
    torch_forward = torch_layer.forward(torch_x)
    torch_backward_y = torch.tensor(backward_y, dtype=float)
    torch_forward.backward(torch_backward_y)
    np.testing.assert_equal(
      torch_forward.detach().numpy(),
      output
    )
    np.testing.assert_almost_equal(
      torch_x.grad.detach().numpy(),
      backward_output,
      decimal=8
    )
  
  def test_random(self):
    for _ in range(5):
      input_channels = random.randint(1,10)
      kernel_size = random.randint(2,6)
      stride = random.randint(1,6)
      output_width = random.randint(3,10)
      output_height = output_width #random.randint(3,10)
      padding = random.randint(
        0,
        min(
          4,
          ((output_width - 1) * stride + kernel_size) // 2 - 1,
          ((output_height - 1) * stride + kernel_size) // 2 - 1,
          kernel_size // 2
        )
      )
      batch_size = random.randint(1,8)
      input_width = (output_width - 1) * stride + kernel_size - 2 * padding
      input_height = (output_height - 1) * stride + kernel_size - 2 * padding
      layer = MaxPool2DLayer(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
      )

      X = np.random.random((batch_size, input_channels, input_width, input_height))
      backward_y = np.random.random((batch_size, input_channels, output_width, output_height))
      output = layer.forward(X)
      backward_output = layer.backward(backward_y)

      torch_layer = get_torch_layer(layer)
      torch_x = torch.tensor(X, dtype=float, requires_grad=True)
      torch_forward = torch_layer.forward(torch_x)
      torch_backward_y = torch.tensor(backward_y, dtype=float)
      torch_forward.backward(torch_backward_y)

      np.testing.assert_equal(
        torch_forward.detach().numpy(),
        output
      )

      np.testing.assert_almost_equal(
        torch_x.grad.detach().numpy(),
        backward_output,
        decimal=8
      )

if __name__ == "__main__":
  unittest.main()