import sys
import load
from random_weights import generate_random_weights
from print_matrices import print_matrices


if len(sys.argv) >= 2:
    network_filename = sys.argv[1]
else:
    print("\nUsage:\t python generate_weights.py network\n")
    sys.exit()

network = load.load_network_structure(network_filename)
weights = generate_random_weights(network['layer_sizes'])

print_matrices(weights)