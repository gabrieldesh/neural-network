import sys
import load
from normalize import normalize_features
from print_matrices import print_matrices
from numeric_verification import verify
import train


if len(sys.argv) >= 4:
    network_filename = sys.argv[1]
    weights_filename = sys.argv[2]
    dataset_name = sys.argv[3]
else:
    print("\nUsage:\t python backpropagation.py network weights dataset\n")
    sys.exit()

dataset = load.load_dataset(dataset_name)
network = load.load_network_structure(network_filename)
initial_weights = load.load_weights(weights_filename)

normalize_features(dataset)

# Calcula gradientes usando todas as instâncias do dataset.
gradients = train.calculate_gradients(dataset, 0, len(dataset), initial_weights, network['regularization'])

print_matrices(gradients)
