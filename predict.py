import numpy as np

def classify_all(dataset, weights):
  classifications = {}
  for i in range(len(dataset)):
    classifications[i] = classify(dataset[i]['input'], weights)
  return classifications

def predict_all(dataset, weights):
  predictions = {}
  for i in range(len(dataset)):
    predictions[i] = predict(dataset[i]['input'], weights)
  return predictions

def classify(input_vector, weights):
  a = propagate(input_vector, weights)[-1]
  for i in range(len(a)):
    if a[i][0] >= 0.5: a[i][0] = 1.0
    else: a[i][0] = 0.0
  return a

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
  