import numpy as np

from src.core.network_base import LossModule, NetworkModule, translate_to_minibatch, softmax, sigmoid

class CrossEntropyLoss(LossModule):
  def __init__(self, model: NetworkModule, reduction_mode: str = 'mean') -> None:
    super().__init__(model)
    self.reduction_mode = reduction_mode

  def __call__(self, predicted_values: np.ndarray, truth_values: np.ndarray):
    predicted_values = translate_to_minibatch(predicted_values)
    if len(truth_values.shape) != 1:
      raise Exception("Classes should have one dimensional")
    if predicted_values.shape[0] != truth_values.shape[0]:
      raise Exception("Classes and predicted values should be the same length")
    if predicted_values.shape[1] < truth_values.max():
      raise Exception("Number of classes should be less than shape of predicted value")
    truth_value = np.zeros_like(predicted_values)
    truth_value[np.arange(truth_values.size), truth_values] = 1
    p = softmax(predicted_values)
    # loss = sum(-1 * truth * log (softmax(pred)))
    loss = np.sum(np.sum(-1 * truth_value * np.log(p),axis=1))
    # grad = (softmax(pred) - truth)
    self.grad = (p - truth_value)
    if self.reduction_mode == 'mean':
      loss /= predicted_values.shape[0]
      self.grad /= predicted_values.shape[0]

    return loss
  
  def backward(self) -> None:
    self.model.backward(self.grad)

class BCEWithLogitsLoss(LossModule):
  def __init__(self, model: NetworkModule, reduction_mode: str = 'mean') -> None:
    super().__init__(model)
    self.reduction_mode = reduction_mode

  def __call__(self, predicted_values: np.ndarray, truth_values: np.ndarray):
    predicted_values = translate_to_minibatch(predicted_values)
    truth_values = translate_to_minibatch(truth_values)
    if predicted_values.shape != truth_values.shape:
      raise Exception("Truth and predicted values should have the same shapes")
    p = sigmoid(predicted_values)
    loss = np.sum(np.sum(-1 * (truth_values * np.log(p) + (1 - truth_values) * np.log(1 - p)),axis=1))
    self.grad = (p - truth_values)
    if self.reduction_mode == 'mean':
      loss /= (predicted_values.shape[1] * predicted_values.shape[0])
      self.grad /= (predicted_values.shape[1] * predicted_values.shape[0])

    return loss
  
  def backward(self) -> None:
    self.model.backward(self.grad)