from predict import predict
import numpy as np

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
  for weight_matrix in weights:
    for i in range(weight_matrix.shape[0]):
      for j in range(1, weight_matrix.shape[1]): # Ignora pesos de bias
        regularization_term += weight_matrix[i, j] ** 2
  regularization_term *= regularization / (2 * len(dataset))

  return mean_error + regularization_term