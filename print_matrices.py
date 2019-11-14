def print_matrices(matrices):
  for matrix in matrices:
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        print(f'{matrix[i, j] :.10f}', end='')
        if j < matrix.shape[1] - 1: print(', ', end='')
      if i < matrix.shape[0] - 1: print('; ', end='')
    print()