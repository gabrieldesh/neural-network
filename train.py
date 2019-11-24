import numpy as np
from predict import propagate
from print_matrices import print_matrices
from cost import J

def calculate_gradients(dataset, start_index, batch_size, weights, regularization):
  # Inicializa os acumuladores de gradientes
  D = []
  for weight_matrix in weights:
    D.append(np.zeros(weight_matrix.shape))

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
  start_index = 0

  # Copia matrizes de initial_weights
  weights = []
  for weight_matrix in initial_weights:
    weights.append(np.array(weight_matrix))

  # Inicializa valores de momento com 0
  z = []
  for weight_matrix in weights:
    z.append(np.zeros(weight_matrix.shape))
  
  # Treina até que ocorram muitas iterações sem melhoria significativa
  best_cost = J(dataset, weights, regularization)
  best_weights = []
  for weight_matrix in weights:
    best_weights.append(np.array(weight_matrix))
  
  num_iterations_without_improvement = 0
  while num_iterations_without_improvement < 10:

    gradients = calculate_gradients(dataset, start_index, batch_size, weights, regularization)

    # Atualiza os pesos
    for k in range(len(gradients)):
      z[k] = momentum * z[k] + gradients[k]
      weights[k] = weights[k] - learning_rate * z[k]
    
    new_cost = J(dataset, weights, regularization)
    if new_cost < best_cost and float(best_cost - new_cost) > float(best_cost*0.001):
      best_cost = new_cost

      best_weights = []
      for weight_matrix in weights:
        best_weights.append(np.array(weight_matrix))

      num_iterations_without_improvement = 0
    else:
      num_iterations_without_improvement += 1

    start_index = (start_index + batch_size) % len(dataset)

  return best_weights