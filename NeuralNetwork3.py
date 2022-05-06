import sys
import time
import datetime
import numpy as np
import pandas as pd
from Layer import Layer

PREDICTIONS_FILE = 'test_predictions.csv'
INPUT_LAYER_SIZE = 784
HIDDEN_LAYER_1_SIZE = 128
HIDDEN_LAYER_2_SIZE = 64
OUTPUT_LAYER_SIZE = 10
EPOCHS = 40
BATCH_SIZE = 32
LR = .2

class Classifier():
  __slots__ = [
    'training_data',
    'training_labels',
    'validation_data',
    'alpha',
    'epochs',
    'batch_size'
  ]
  def __init__(
    self,
    training_data,
    training_labels,
    validation_data,
    learning_rate=LR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
  ):
    self.training_data = training_data
    self.training_labels = training_labels
    self.validation_data = validation_data
    self.alpha = learning_rate
    self.epochs = epochs
    self.batch_size = batch_size
 
  def create_network(self):
    # Create layers
    output_layer = Layer(HIDDEN_LAYER_2_SIZE, OUTPUT_LAYER_SIZE, output=True)
    hidden_layer_2 = Layer(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, next=output_layer)
    hidden_layer_1 = Layer(INPUT_LAYER_SIZE, HIDDEN_LAYER_1_SIZE, next=hidden_layer_2)
    hidden_layer_2.set_prev(hidden_layer_1)
    output_layer.set_prev(hidden_layer_2)
    return (hidden_layer_1, hidden_layer_2, output_layer)

  def forward_pass(self, inputs, network):
    pass_result = inputs
    for layer in network:
      output_vector = layer.compute_output_vector(pass_result)
      output = layer.activation_function(output_vector)
      pass_result = output
    return pass_result

  # cross-entropy loss function
  def calculate_loss_gradient(self, output, expected):
    gradient_loss = output - expected
    return gradient_loss

  def get_one_hot(self, shape, expected):
    one_hot = np.zeros(shape)
    np.put_along_axis(one_hot, expected, 1, axis=1)
    return one_hot

  def backpropagate(self, network, loss):
    for layer in network[::-1]:
      layer.update_deltas(loss)
      layer.update_weights(self.alpha)

  def shuffle_and_batch_data(self, images, labels):
    training_set_size = len(images) // 6
    random_order = np.random.permutation(training_set_size)
    random_imgs = images[random_order][0:10000]
    random_labels = labels[random_order][0:10000]
    split_size = training_set_size if training_set_size < self.batch_size else training_set_size // self.batch_size

    return np.array_split(random_imgs, split_size), np.array_split(random_labels, split_size)

  def get_accuracy(self, predictions):
    return np.mean(predictions)

  def preprocess_data(self, inputs):
    return (inputs / 255).astype('float32')
  
  def train(self, network):
    print('Reading training data...')
    data_csv = pd.read_csv(self.training_data, header=None)
    print('Preprocessing inputs...')
    data_csv = self.preprocess_data(data_csv)
    print('Reading training labels...')
    label_csv = pd.read_csv(self.training_labels, header=None)
    print('Training with sample of size', data_csv.shape)
    for epoch in range(1, self.epochs + 1):
      epoch_predictions = []
      shuffled_img, shuffled_label = self.shuffle_and_batch_data(data_csv.to_numpy(), label_csv.to_numpy())
      for batch_data, batch_label in zip(shuffled_img, shuffled_label):
        if len(batch_data) > 0:
          output = self.forward_pass(batch_data, network)
          one_hot = self.get_one_hot(output.shape, batch_label)
          epoch_predictions = np.append(epoch_predictions, np.argmax(output, axis=1) == np.argmax(one_hot, axis=1))
          loss = self.calculate_loss_gradient(output, one_hot)
          self.backpropagate(network, loss)
      
      print('After', str(epoch) + ':', f'{(self.get_accuracy(epoch_predictions) * 100):.5f}%')
      epoch_predictions = []
      self.decay_lr(epoch)

    return network

  def decay_lr(self, epoch):
    decay_rate = .01
    self.alpha = LR / (1 + decay_rate * epoch)

  def validate(self, network):
    print('Reading test data...')
    validation_csv = pd.read_csv(test_data_url, header=None)
    validation_csv = self.preprocess_data(validation_csv)
    final_outputs = self.forward_pass(validation_csv.to_numpy(), network)
    predictions = np.argmax(final_outputs, axis=1)
    return pd.Series(predictions)

if __name__ == '__main__':
  _, train_data_url, train_labels_url, test_data_url, *test_labels_url = sys.argv
  classifier = Classifier(training_data=train_data_url, training_labels=train_labels_url, validation_data=test_data_url)
  start = time.time()
  print('Creating network...')
  network = classifier.create_network()
  network = classifier.train(network)
  print('Trained!')
  print('Validating')
  predictions = classifier.validate(network)
  print('Validated')
  print('Writing to:', PREDICTIONS_FILE)

  print('Completion time: ', str(datetime.timedelta(seconds=(time.time() - start))))
  if len(sys.argv) > 4:
    # For testing, a validation parameter to compare predicted results with actual results
    validation_labels = pd.read_csv(test_labels_url[0], header=None)
    final_preds = predictions.to_numpy() == validation_labels.to_numpy().flatten()
    print('Validation accuracy:', f'{(classifier.get_accuracy(final_preds) * 100):.5f}%')
    

  predictions.to_csv(PREDICTIONS_FILE, index=False, header=None)


