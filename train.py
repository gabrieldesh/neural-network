import numpy as np
from predict import propagate

def calculate_gradients(dataset, start_index, batch_size, weights, regularization):
  # Inicializa os acumuladores de gradientes
  D = []
  for matrix in weights:
    D.append(np.zeros(matrix.shape))

  # Calcula os gradientes para cada instância
  for i in range(start_index, start_index + batch_size):
    instance = dataset[i % len(dataset)]
    x = instance['input']
    y = instance['output']
    a = propagate(x, weights)
    fx = a[-1] # Ativações da camada de saída
    delta = [None] * len(a)

    # Calcula deltas da camada de saída
    delta[-1] = fx - y

    # Calcula deltas para as camadas ocultas
    for k in range(len(weights) - 1, 0, -1):
      delta[k] = weights[k].T @ delta[k + 1] * a[k] * (1 - a[k])
      # Remove o primeiro elemento (delta de bias)
      delta[k] = np.delete(delta[k], 0, 0)
    
    # Calcula os gradientes e acumula em D
    for k in range(len(D)):
      D[k] = D[k] + delta[k + 1] @ a[k].T
  
  # Aplica regularização aos gradientes
  for k in range(len(D)):
    P = regularization * weights[k]
    P[:, 0] = 0 # Zera a primeira coluna
    D[k] = (D[k] + P) / batch_size
  
  return D


def backpropagation(dataset, initial_weights, regularization, learning_rate, momentum, batch_size):
  # TODO Usar a calculate_gradients repetidamente para treinar a rede neural em mini-batches, até atingir um
  # critério de parada.
  return None