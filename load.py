import json
import pandas

def load_dataset(name):
  """ name: Nome do dataset, p.ex. 'credit-g'
  """

  # Lê os metadados
  with open('metadata/{}.json'.format(name), 'r') as f:
      metadata = json.load(f)
  
  # Lê os dados
  data = pandas.read_csv('datasets/{}.csv'.format(name))

  # Cria um dicionário de atributos, e descobre o atributo alvo
  target = ""
  attributes = {}
  for feature in metadata['features']:
    attribute = {}
    attribute['type'] = feature['type']
    if 'distr' in feature:
      attribute['values'] = feature['distr'][0]
    if 'target' in feature and feature['target'] == "1":
      target = feature['name']
    attributes[feature['name']] = attribute

  # Um dataset é um dicionário com dados e informações dos atributos.
  # 'data' é um pandas DataFrame.
  # 'attributes' é uma lista de dicionários. Cada dicionário contém uma série de informações de um atributo. As mais
  # importantes são:
  #   ['name']: Nome do atributo.
  #   ['type']: Tipo do atributo ('nominal' ou 'numeric').
  #   ['values']: Presente somente se o atributo for nominal. Lista de valores possíveis do atributo.
  # 'target' é uma string, o nome do atributo alvo.
  return {
    'data': data,
    'attributes': attributes,
    'target': target
  }

def load_network_structure(filename):
  # TODO
  return {
    'regularization': 0.0,
    'layer_sizes': [1, 2, 1]
  }

def load_weights(filename):
  # TODO: Retornar matrizes de pesos para vetorização.
  return None