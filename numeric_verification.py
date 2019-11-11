from predict import predict
import numpy as np

def verify(dataset, network_structure, initial_weights):
  epsilon = 0.01
  gradients = []

  for theta in initial_weights:
    gradient_matrix = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
      for j in range(theta.shape[1]):
        v = theta[i, j]

        theta[i, j] = v - epsilon
        J0 = J(dataset, network_structure, initial_weights)

        theta[i, j] = v + epsilon
        J1 = J(dataset, network_structure, initial_weights)

        # Restaura o valor do peso
        theta[i, j] = v

        gradient_matrix[i, j] = (J1 - J0) / 2 * epsilon
    gradients.append(gradient_matrix)
  
  return gradients

def J(dataset, network_structure, weights):
  sum = 0
  for instance in dataset:
    x = instance['input']
    y = instance['output']
    fx = predict(x, network_structure, weights)
    errors = -y * np.log(fx) - (1 - y) * np.log(1 - fx)
    sum += np.sum(errors) # Soma erros de cada saída.
    
  return sum
