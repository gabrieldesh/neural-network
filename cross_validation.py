import pandas
import math
import copy

def cross_validation(dataset, k):
    
    dlen = len(dataset)
    num_outputs = len(dataset[0]['output'])
    kfolds = {}

    # cria k folds
    for i in range(k):
        kfolds[i] = []

    # CASO 1: só há 1 output, e ele é zero ou um
    if num_outputs == 1:

        # divide o conjunto original em 2 classes: 0 e 1
        class0 = []
        class1 = []
        #print(dlen)
        for j in range(dlen):
            if dataset[j]['output'][0][0] == 1.0:
                class1.append(dataset[j])
            else:
                class0.append(dataset[j])
        #print(len(class0))
        #print(len(class1))        
        
        # coloca len(class1)/k instancias com output=1 em cada fold
        # coloca len(class0)/k instancias com output=0 em cada fold
        numk = 0
        for i in range(len(class1)):
            if numk == k: numk = 0
            kfolds[numk].append(class1[i])
            numk += 1
        for i in range(len(class0)):
            if numk == k: numk = 0
            kfolds[numk].append(class0[i])
            numk += 1

        """for i in range(k): # DEBUG CODE
            freq0count = 0
            freq1count = 0
            for j in range(len(kfolds[i])):
                if kfolds[i][j]['output'][0][0] == 1.0:
                    freq1count += 1
                else:
                    freq0count += 1
            print('fold {} counts'.format(i))
            print(freq0count)
            print(freq1count)
            print(f'Total: {len(kfolds[i])}\n')"""
            
        return kfolds

    # CASO 2: há mais de 1 output, deve-se preservar a quantidades de instancias "1" de cada output em cada fold
    elif num_outputs > 1:

        #divide o conjunto original em 'num_outputs' classes
        classes = {}
        for i in range(num_outputs):
            classes[i] = []
        for i in range(dlen):
            for j in range(num_outputs):
                if dataset[i]['output'][j][0] == 1.0:
                    classes[j].append(dataset[i])
        
        numk = 0
        for i in range(num_outputs):

            # coloca len(classes[i])/k instâncias em cada fold
            for j in range(len(classes[i])):
                if numk == k: numk = 0
                kfolds[numk].append(classes[i][j])
                numk += 1

        """for i in range(k): # DEBUG CODE
            freqcount = {}
            for j in range(num_outputs):
                freqcount[j] = 0
            for j in range(len(kfolds[i])):
                for k in range(num_outputs):
                    if kfolds[i][j]['output'][k][0] == 1.0:
                        freqcount[k] += 1
            print('fold {} counts'.format(i))
            for j in range(num_outputs):
                print(freqcount[j])
            print(f'Total: {len(kfolds[i])}\n')"""

        return kfolds

    print('kFolds successfully generated.')