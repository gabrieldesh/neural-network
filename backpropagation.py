import sys
import load
from normalize import normalize_features
from print_gradients import print_gradients
import train


if len(sys.argv) >= 4:
    network_filename = sys.argv[1]
    weights_filename = sys.argv[2]
    dataset_name = sys.argv[3]
else:
    print("\nUsage:\t python backpropagation.py network weights dataset_name\n")
    sys.exit()

dataset = load.load_dataset(dataset_name)
network_structure = load.load_network_structure(network_filename)
initial_weights = load.load_weights(weights_filename)

dataset = normalize_features(dataset)

# Calcula gradientes usando todas as instâncias do dataset.
dataset_size = 0 # TODO: Calcular tamanho do dataset.
gradients = train.calculate_gradients(dataset, dataset_size, 0, network_structure, initial_weights)

print_gradients(gradients)