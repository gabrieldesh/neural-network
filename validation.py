import sys
import f1measure as f1
import f1measure2 as f1_2
import random
import numpy
import load
import cost
import cross_validation as cv
from train import backpropagation
from random_weights import generate_random_weights
from normalize import normalize_features


if len(sys.argv) >= 3:
    network_filename = sys.argv[1]
    dataset_name = sys.argv[2]
    learning_rate = 1.0
    if len(sys.argv) >= 4:
        learning_rate = float(sys.argv[3])
    momentum = 1.0
    if len(sys.argv) >= 5:
        momentum = float(sys.argv[4])
    batch_size = 20
    if len(sys.argv) >= 6:
        batch_size = int(sys.argv[5])
    k = 10
    if len(sys.argv) >= 7:
        k = int(sys.argv[6])
    seed = 0
    if len(sys.argv) >= 8:
        seed = int(sys.argv[7])
else:
    print("\nUsage:\t python validation.py network dataset [learning_rate] [momentum] [batch_size] [k] [seed]\n")
    sys.exit()

numpy.random.seed(seed)

dataset = load.load_dataset(dataset_name)
network = load.load_network_structure(network_filename)
initial_weights = generate_random_weights(network['layer_sizes'])

print("Entradas:")
print(f"dataset: {dataset_name}")
print(f"layer sizes: {network['layer_sizes']}")
print(f"regularization: {network['regularization']}")
print(f"learning rate: {learning_rate}")
print(f"momentum: {momentum}")
print(f"batch size: {batch_size}")
print(f"num. folds (k): {k}")
print(f"random seed: {seed}")

if network['layer_sizes'][len(network['layer_sizes'])-1] == 1:
    confusion_matrices = f1.eval_confusion_matrices(dataset, initial_weights, network['regularization'],
                                                 learning_rate, momentum, batch_size, k)
    f1.eval_f1measure(confusion_matrices)
else:
    confusion_matrices = f1_2.eval_confusion_matrices(dataset, initial_weights, network['regularization'],
                                                 learning_rate, momentum, batch_size, k)
    f1_2.eval_f1measure(confusion_matrices)