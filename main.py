import sys
import f1measure as f1
import random
import numpy
import load
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
    batch_size = 50
    if len(sys.argv) >= 6:
        batch_size = int(sys.argv[5])
    k = 5
    if len(sys.argv) >= 7:
        k = int(sys.argv[6])
    seed = None
    if len(sys.argv) >= 8:
        seed = int(sys.argv[7])
    posclass = None
    if len(sys.argv) >= 9:
        posclass = sys.argv[8]
else:
    print("\nUsage:\t python main.py network dataset [learning_rate] [momentum] [batch_size] [k] [seed] "
          "[positive_class_value]\n")
    sys.exit()

if (seed != None):
    random.seed(seed)
    numpy.random.seed(seed)

dataset = load.load_dataset(dataset_name)
network_structure = load.load_network_structure(network_filename)
initial_weights = generate_random_weights(network_structure)

normalize_features(dataset)

confusion_matrices = f1.eval_confusion_matrices(dataset, network_structure, initial_weights, 
                                                learning_rate, momentum, batch_size, k)
f1.eval_f1measure(confusion_matrices, posclass)