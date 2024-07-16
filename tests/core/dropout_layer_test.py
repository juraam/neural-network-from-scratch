import unittest
import numpy as np

import torch

from src.core.dropout_layer import DropoutLayer

class DropoutLayerTest(unittest.TestCase):
  def test_raise_exception_when_p_1(self):
    self.assertRaises(Exception, DropoutLayer, 1)
  
  def test_raise_exception_when_p_greater_1(self):
    self.assertRaises(Exception, DropoutLayer, 2)

  def test_raise_exception_when_p_0(self):
    self.assertRaises(Exception, DropoutLayer, 0)

  def test_raise_exception_when_p_less_0(self):
    self.assertRaises(Exception, DropoutLayer, -2)

  def test_single_row(self):
    p = 0.5
    layer = DropoutLayer(p=p)

    X = np.array([1,1,1,1,1])
    backward_y = np.array([1, 1, 1, 1, 1])
    output = layer.forward(X)
    output_backward = layer.backward(backward_y)

    np.testing.assert_equal(
      layer.mask.shape,
      X.shape
    )
    self.assertEqual((output > 0).sum(), (output_backward > 0).sum())
  
  def test_batch(self):
    layer = DropoutLayer()

    X = np.random.random((5,20,20,20))
    backward_y = np.random.random((5,20,20,20))
    output = layer.forward(X)
    output_backward = layer.backward(backward_y)

    np.testing.assert_equal(
      layer.mask.shape,
      X.shape
    )
    self.assertEqual((output > 0).sum(), (output_backward > 0).sum())

if __name__ == "__main__":
  unittest.main()