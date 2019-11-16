import math
def normalize_features(dataset):
  # Encontra o mínimo e máximo de cada atributo.
  mins = list(dataset[0]['input'].T[0])
  maxs = list(dataset[0]['input'].T[0])  
  num_attrs = len(mins)

  for i in range(1, len(dataset)):
    for j in range(0, num_attrs):
      mins[j] = min(mins[j], dataset[i]['input'][j, 0])      
      maxs[j] = max(maxs[j], dataset[i]['input'][j, 0]) 
  
  # Normaliza os valores dos atributos.
  for instance in dataset:
    x = instance['input']
    for j in range(0, num_attrs):
      if math.isnan(mins[j]): print('min {}'.format(mins[j]))
      if math.isnan(maxs[j]): print('max {}'.format(maxs[j]))
      if (maxs[j] - mins[j]) == 0.0: x[j, 0] = 0.0
      else: x[j, 0] = (x[j, 0] - mins[j]) / (maxs[j] - mins[j])