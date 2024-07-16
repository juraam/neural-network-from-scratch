import unittest
import numpy as np
import random

import torch

from src.core.flatten_layer import FlattenLayer

class FlattenLayerTest(unittest.TestCase):
  def test_batch(self):
    layer = FlattenLayer()

    X = np.array([[ [ [1,2], [3,4] ], [ [5,6], [7, 8] ], [ [9,10], [11, 12] ] ] ])
    output = layer.forward(X)
    output_backward = layer.backward(output)

    np.testing.assert_equal(
      output,
      np.array([[1,2,3,4,5,6,7,8,9,10,11,12]])
    )
    np.testing.assert_equal(
      X,
      output_backward
    )

    torch_layer = torch.nn.Flatten()
    torch_x = torch.tensor(X, dtype=float, requires_grad=True)
    torch_output = torch_layer.forward(torch_x)
    torch_forward = torch_layer.forward(torch_x)
    torch_forward.backward(torch_output)
    np.testing.assert_equal(
      torch_x.grad.detach().numpy(),
      output_backward
    )
    np.testing.assert_equal(
      torch_output.detach().numpy(),
      output
    )

  def test_random_batch(self):
    layer = FlattenLayer()
    batch = random.randint(1, 20)
    shape = random.randint(1, 6)
    shape_array = []
    flatten_shape = 1
    for i in range(shape):
      number = random.randint(1, 5)
      flatten_shape *= number
      shape_array.append(number)
    shape_array = [batch] + shape_array
    X = np.random.random(shape_array)
    backward_y = np.random.random((batch, flatten_shape))
    output = layer.forward(X)
    output_backward = layer.backward(backward_y)

    torch_layer = torch.nn.Flatten()
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