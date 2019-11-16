import sys
import f1measure as f1
import random
import numpy
import load
from train import backpropagation
from random_weights import generate_random_weights
from normalize import normalize_features
import cross_validation as cv
import predict

dataset = load.load_dataset("datasets/wine.txt")
network = load.load_network_structure("networks/wine_0.txt")
iw = generate_random_weights(network['layer_sizes'])
cvset = cv.cross_validation(dataset, 10)
w = backpropagation(cvset[0]['trainingSet'],iw, network['regularization'],1.0,0.8,50)
classif = predict.classify_all(cvset[0]['testSet'],w)