import pandas
import math
import copy

def cross_validation(dataset, k):
    partitions = []
    kFolds = []
    training_dict = []
    test_dict = []
    fold_dict = []

    data = dataset['data']

    len_data = len(data)
    #Define o número de instâncias em cada fold
    instancesPerFold = len_data/k
    
    if instancesPerFold < 2:
        raise Exception("Quantidade de folds muito grande para o conjunto de dados")
    
    print('Generating kFolds...')
    
    instancesPerFold = round(instancesPerFold)
    
    #Ordena o dataset - onde columns é a classe e count é a frequencia de cada instancia
    columns = dataset['target']
    count = data[columns].value_counts()
    data = data.sort_values(by=columns)
    count = count.sort_index()
    count = count.tolist()
    
    start = 0
    end = len_data
    aux = []
    #Divide o dataset no numero de classes onde cada classe n possui n/k instancias 
    for j in range(len(count)):
        instances = int(round(count[j]/k))
        aux = [data[i:i+instances] for i in range(start, end, instances)]
        start = start + count[j]
        partitions.append(aux)
            
    
    trainingSet = []
    fold = []
    #Tenta acessar cada particao pra pegar o primeiro de cada classe e formar um fold
    for i in range(k):
        testSet = []

        #Insere o primeiro de cada partition pra fazer o cross-validation estratificado 
        for j in  range(len(partitions)):
            testSet.append(partitions[j][i])

        #Copia o dataset para trainingSet   
        trainingSet = copy.deepcopy(partitions[0])
        #Concatena para normalizar 
        testSet = pandas.concat(testSet) 
        trainingSet = pandas.concat(trainingSet)
        #Exclui o conjunto de dados usado por testSet 
        trainingSet = trainingSet[~trainingSet.apply(tuple,1).isin(testSet.apply(tuple,1))] 

        trainingSet = trainingSet.reset_index(drop=True)
        testSet = testSet.reset_index(drop=True)

        training_dict = {
            'data': trainingSet,
            'attributes': dataset['attributes'],
            'target': dataset['target']
        }
        
        test_dict = {
            'data': testSet,
            'attributes': dataset['attributes'],
            'target': dataset['target']
        }
        
        fold_dict = {
            'trainingSet': training_dict,
            'testSet': test_dict
        }

        #print("testSet:\n",testSet)
        #print("trainingSet:\n", trainingSet)
        
        kFolds.insert(k, fold_dict)      
        
    

    #print(kFolds)
    print('kFolds successfully generated.')
    return kFolds