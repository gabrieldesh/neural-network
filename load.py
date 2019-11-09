import json
import numpy as np

def load_dataset(filename):
  dataset = []
  with open(filename, 'r') as f:
    line = f.readline()
    while line:
      [input_string, output_string] = line.split(';')
      input_vector = np.array(np.mat(input_string).T)
      output_vector = np.array(np.mat(output_string).T)
      dataset.append({
        'input': input_vector,
        'output': output_vector
      })
      line = f.readline()
    
  return dataset


def load_network_structure(filename):
  # TODO
  return {
    'regularization': 0.0,
    'layer_sizes': [1, 2, 1]
  }


def load_weights(filename):
  # TODO: Retornar matrizes de pesos para vetorização.
  return None