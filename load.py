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
  with open(filename, 'r') as f:
    regularization = float(f.readline())

    layer_sizes = []
    line = f.readline()
    while line:
      layer_sizes.append(int(line))
      line = f.readline()

  return {
    'regularization': regularization,
    'layer_sizes': layer_sizes
  }


def load_weights(filename):
  with open(filename, 'r') as f:
    weight_matrices = []
    line = f.readline()
    while line:
      matrix = np.array(np.mat(line))
      weight_matrices.append(matrix)
      line = f.readline()
  
  return weight_matrices