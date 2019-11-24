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


def backpropagation(training_set, test_set, initial_weights, regularization, learning_rate, momentum, batch_size, 
                    print_results = False):
  start_index = 0

  # Copia matrizes de initial_weights
  weights = []
  for weight_matrix in initial_weights:
    weights.append(np.array(weight_matrix))

  # Inicializa valores de momento com 0
  z = []
  for weight_matrix in weights:
    z.append(np.zeros(weight_matrix.shape))
  
  best_cost = J(test_set, weights, regularization)
  best_weights = []
  for weight_matrix in weights:
    best_weights.append(np.array(weight_matrix))
  
  costs = [best_cost]

  # Treina até que ocorram muitas iterações sem melhoria significativa
  num_iterations_without_improvement = 0
  while num_iterations_without_improvement < 10:

    gradients = calculate_gradients(training_set, start_index, batch_size, weights, regularization)

    # Atualiza os pesos
    for k in range(len(gradients)):
      z[k] = momentum * z[k] + gradients[k]
      weights[k] = weights[k] - learning_rate * z[k]
    
    new_cost = J(test_set, weights, regularization)
    costs.append(new_cost)
    if new_cost < best_cost:
      # Considera apenas melhorias significativas
      if float(best_cost - new_cost) > 1e-5:
        num_iterations_without_improvement = 0
      else:
        num_iterations_without_improvement += 1
      
      best_cost = new_cost

      best_weights = []
      for weight_matrix in weights:
        best_weights.append(np.array(weight_matrix))
    else:
      num_iterations_without_improvement += 1

    start_index = (start_index + batch_size) % len(training_set)

  if print_results:
    print("Treinamento concluído. Custos:")
    for cost in costs:
      print(cost)

    print(f"\nMelhor custo encontrado: {best_cost}")
    print("Melhores pesos encontrados:")
    print_matrices(best_weights)

  return best_weights