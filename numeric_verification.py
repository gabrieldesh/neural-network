from cost import J
import numpy as np

def verify(dataset, initial_weights, regularization):
  epsilon = 0.000001
  gradients = []

  for theta in initial_weights:
    gradient_matrix = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
      for j in range(theta.shape[1]):
        v = theta[i, j]

        theta[i, j] = v - epsilon
        J0 = J(dataset, initial_weights, regularization)

        theta[i, j] = v + epsilon
        J1 = J(dataset, initial_weights, regularization)

        # Restaura o valor do peso
        theta[i, j] = v

        gradient_matrix[i, j] = (J1 - J0) / (2 * epsilon)
    gradients.append(gradient_matrix)
  
  return gradients