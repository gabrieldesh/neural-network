import numpy as np

def generate_random_weights(layer_sizes):
  weights = []
  for k in range(len(layer_sizes) - 1):
    matrix_size = (layer_sizes[k + 1], layer_sizes[k] + 1)
    weights.append(np.random.standard_normal(matrix_size))
  
  return weights