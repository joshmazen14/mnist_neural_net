import numpy as np
from scipy.special import expit, softmax

class Layer:
  __slots__ = [
    'weights',
    'bias',
    'next',
    'prev',
    'is_output',
    'inputs',
    'output_vector',
    'delta_output',
    'delta_weight',
    'delta_bias'
  ]

  def __init__(self, input_size, num_neurons, next=None, prev=None, output=False):
    self.weights = self.initialize_weights(input_size, num_neurons)
    self.bias = self.initialize_biases(num_neurons)
    self.prev = prev
    self.next = next
    self.is_output = output

  def set_prev(self, prev):
    self.prev = prev

  def initialize_weights(self, from_layer, to_layer):
    return np.random.normal(scale=(1 / from_layer) ** .5 , size=(from_layer, to_layer))

  def initialize_biases(self, size):
    return np.zeros((1, size))

  def update_deltas(self, loss):
    batch_size = self.output_vector.shape[0]
    act_prime = 1 if self.is_output else self.activation_derivative(self.output_vector).T
    output_partial = loss.T if self.is_output else self.next.weights @ self.next.delta_output
    prev_out = self.inputs if not self.prev else self.prev.output_vector
    self.delta_output = output_partial * act_prime
    self.delta_weight = (self.delta_output @ prev_out) / batch_size
    self.delta_bias = np.sum(self.delta_output, axis=1) / batch_size


  def update_weights(self, rate):
    self.weights -= rate * self.delta_weight.T
    self.bias -= rate * self.delta_bias

  # Computes the vector that feeds into the next layer
  def compute_output_vector(self, inputs):
    self.inputs = inputs
    self.output_vector = (inputs @ self.weights) + self.bias
    return self.output_vector
  
  # Takes a vector x and returns the activation function
  # Note: expit = sigmoid fn
  def activation_function(self, x):
    activation_bound = 500
    clipped_x = np.clip(x, -activation_bound, activation_bound) # clip to prevent gradient explosion or vanishing
    return softmax(x, axis=1) if self.is_output else expit(clipped_x)

  def activation_derivative(self, x):
    return self.activation_function(x) * (1 - self.activation_function(x))