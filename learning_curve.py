import sys
import f1measure as f1
import random
import numpy
import load
from train import backpropagation
from random_weights import generate_random_weights
from normalize import normalize_features
from cross_validation import holdout


if len(sys.argv) >= 3:
    network_filename = sys.argv[1]
    dataset_name = sys.argv[2]
    learning_rate = 1.0
    if len(sys.argv) >= 4:
        learning_rate = float(sys.argv[3])
    momentum = 1.0
    if len(sys.argv) >= 5:
        momentum = float(sys.argv[4])
    batch_size = 1
    if len(sys.argv) >= 6:
        batch_size = int(sys.argv[5])
    seed = 0
    if len(sys.argv) >= 7:
        seed = int(sys.argv[6])
else:
    print("\nUsage:\t python train_single_network.py network dataset [learning_rate] [momentum] [batch_size] "
          "[seed]\n")
    sys.exit()

numpy.random.seed(seed)

dataset = load.load_dataset(dataset_name)
network = load.load_network_structure(network_filename)
initial_weights = generate_random_weights(network['layer_sizes'])

normalize_features(dataset)

print("Entradas:")
print(f"dataset: {dataset_name}")
print(f"layer sizes: {network['layer_sizes']}")
print(f"regularization: {network['regularization']}")
print(f"learning rate: {learning_rate}")
print(f"momentum: {momentum}")
print(f"batch size: {batch_size}")
print(f"random seed: {seed}")

holdout_sets = holdout(dataset)

print(f"Iniciando treinamento")
backpropagation(holdout_sets['trainingSet'], holdout_sets['testSet'], initial_weights, 
                network['regularization'], learning_rate, momentum, batch_size, print_results = True)