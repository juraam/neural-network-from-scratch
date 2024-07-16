from src.core.network_base import softmax, translate_to_minibatch, sigmoid
import unittest
import numpy as np

class TranslateMinibatchTest(unittest.TestCase):
  def test_when_scalar(self):
    batch = translate_to_minibatch(np.array([0]))
    np.testing.assert_equal(
      batch,
      np.array([[0]])
    )

  def test_when_vector(self):
    batch = translate_to_minibatch(np.array([0,0,0,0,0]))
    np.testing.assert_equal(
      batch,
      np.array([[0,0,0,0,0]])
    )

class SoftmaxTest(unittest.TestCase):
  def test_when_scalar(self):
    result = softmax(np.array([1]))
    np.testing.assert_equal(
      result,
      np.array([[1]])
    )

  def test_when_single_vector(self):
    result = softmax(np.array([1, 2]))
    np.testing.assert_almost_equal(
      result,
      np.array([[0.2689, 0.7311]]),
      decimal=4
    )

  def test_when_multiple_vector(self):
    result = softmax(np.array([[1, 2], [2, 4]]))
    np.testing.assert_almost_equal(
      result,
      np.array([
        [0.2689, 0.7311],
        [0.1192, 0.8808]
      ]),
      decimal=4
    )

  def test_when_large_numbers(self):
    result = softmax(np.array([
      [1001,1002],
      [3,4]
    ]))
    np.testing.assert_almost_equal(
      result,
      np.array([
        [0.2689, 0.7311],
        [0.2689, 0.7311]
      ]),
      decimal=4
    )

class SigmoidTest(unittest.TestCase):
  def test_when_scalar(self):
    result = sigmoid(np.array([1]))
    np.testing.assert_almost_equal(
      result,
      np.array([0.73105858]),
      decimal=8
    )

  def test_when_single_vector(self):
    result = sigmoid(np.array([1, 2]))
    np.testing.assert_almost_equal(
      result,
      np.array([0.73105858, 0.88079708]),
      decimal=8
    )

  def test_when_multiple_vector(self):
    result = sigmoid(np.array([[1, 2], [2, 4]]))
    np.testing.assert_almost_equal(
      result,
      np.array([
        [0.73105858, 0.88079708],
        [0.88079708, 0.98201379]
      ]),
      decimal=8
    )

if __name__ == "__main__":
  unittest.main()