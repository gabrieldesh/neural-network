import numpy as np

def predict_all(dataset, network_structure, weights):
  # TODO Retornar uma lista com as predições para cada instância do dataset
  return None

def predict(input_vector, network_structure, weights):
  a = input_vector
  for theta in weights:
    a = np.insert(a, 0, 1.0, axis=0) # Adiciona ativação de bias
    z = theta @ a
    a = 1 / (1 + np.exp(-z))
  return a
  