import numpy as np

def predict_all(dataset, weights):
  # TODO Retornar uma lista com as predições para cada instância do dataset
  return None

def predict(input_vector, weights):
  # Retorna as ativações da camada de saída.
  return propagate(input_vector, weights)[-1]

# Propaga um exemplo pela rede. Retorna lista com as ativações de cada camada. A primeira ativação de cada 
# camada é a de bias (1.0), exceto na última camada, de saída.
def propagate(input_vector, weights):
  activations = []
  a = input_vector
  for theta in weights:
    a = np.insert(a, 0, 1.0, axis=0) # Adiciona ativação de bias
    activations.append(a)
    z = theta @ a
    a = 1 / (1 + np.exp(-z))
  activations.append(a)
  return activations
  