from predict import predict
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

def J(dataset, weights, regularization):
  sum = 0
  for instance in dataset:
    x = instance['input']
    y = instance['output']
    fx = predict(x, weights)
    errors = -y * np.log(fx) - (1 - y) * np.log(1 - fx)
    sum += np.sum(errors) # Soma erros de cada saída.  
  mean_error = sum / len(dataset)

  regularization_term = 0
  for matrix in weights:
    for i in range(matrix.shape[0]):
      for j in range(1, matrix.shape[1]): # Ignora pesos de bias
        regularization_term += matrix[i, j] ** 2
  regularization_term *= regularization / (2 * len(dataset))

  return mean_error + regularization_term