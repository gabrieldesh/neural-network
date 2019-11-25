import cross_validation as cv
import statistics
from train import backpropagation
from predict import classify_all

def eval_confusion_matrices(dataset, initial_weights, regularization,
                            learning_rate, momentum, batch_size, k):

    cvset = cv.cross_validation(dataset,k)
    confusion_matrices = {}
    num_outputs = len(cvset[0]['testSet'][0]['output'])        
    for i in range(num_outputs):
        confusion_matrices[i] = {
            'vp': 0,
            'fp': 0,
            'vn': 0,
            'fn': 0,
            'sum': 0
        }
    for i in range(k):
        w = backpropagation(cvset[i]['trainingSet'], cvset[i]['testSet'], initial_weights,
                            regularization, learning_rate, momentum, batch_size)
        classifications = classify_all(cvset[i]['testSet'], w)        
        num_test_instances = len(cvset[i]['testSet'])        
        for j in range(num_test_instances):
            for k in range(num_outputs):
                confusion_matrices[k]['sum'] += 1
                if cvset[i]['testSet'][j]['output'][k][0] == 1.0:
                    if classifications[j][k] == 1.0: confusion_matrices[k]['vp'] += 1
                    else: confusion_matrices[k]['fn'] += 1
                else:
                    if classifications[j][k] == 1.0: confusion_matrices[k]['fp'] += 1
                    else: confusion_matrices[k]['vn'] += 1
    return confusion_matrices
        
                
def eval_f1measure(confusion_matrices, positive_class = None):

    cm = confusion_matrices
    hits = 0

    for key in cm:
        hits += cm[key]['vp']

    results = {
        'acc': [],
        'rec': [],
        'prec': [],
        'spec': [],
        'f1m': []
    }

    stdev = {
        'acc': 0,
        'rec': 0,
        'prec': 0,
        'spec': 0,
        'f1m': 0
    }

    for key in cm:        
        vp = cm[key]['vp']
        fp = cm[key]['fp']
        vn = cm[key]['vn']
        fn = cm[key]['fn']
        summ = cm[key]['sum']
        recall = 0.0
        precision = 0.0
        specificity = 0.0    
        accuracy = float(hits) / float(summ)
        if vp + fn > 0.0:
            recall = float(vp) / float(vp + fn)
        if vp + fp > 0.0:
            precision = float(vp) / float(vp+fp)
        if vn + fn > 0.0:
            specificity = float(vn) / float(vn+fp)
        #if precision + recall > 0.0:
        #    f1m = float(2 * float(precision * recall) / float(precision + recall))
        #print('Matriz de confusão: VP {}, FP {}, VN {}, FN {}'.format(vp,fp,vn,fn))
        results['acc'].append(accuracy)
        results['rec'].append(recall)
        results['prec'].append(precision)
        results['spec'].append(specificity)
        try:
            results['f1m'].append(float(2 * float(precision * recall) / float(precision + recall)))
        except ZeroDivisionError:
            results['f1m'].append(0)
    #Calcula os desvios
    stdev['acc'] = statistics.stdev(results['acc'])
    stdev['rec'] = statistics.stdev(results['rec'])
    stdev['prec'] = statistics.stdev(results['prec'])
    stdev['spec'] = statistics.stdev(results['spec'])
    stdev['f1m'] = statistics.stdev(results['f1m'])

    #Calcula as medias
    results['acc'] = statistics.mean(results['acc'])
    results['rec'] = statistics.mean(results['rec'])
    results['prec'] = statistics.mean(results['prec'])
    results['spec'] = statistics.mean(results['spec'])
    results['f1m'] = statistics.mean(results['f1m'])

    #Imprime os resultados
    print('\nResultados:')
    print('\nF1 measure: {}'.format(results['f1m']))
    print('Acurácia: {}'.format(results['acc']))
    print('Sensibilidade (recall): {}'.format(results['rec']))
    print('Precisão: {}'.format(results['prec']))
    print('Especificidade: {}'.format(results['spec']))
    print('\nDesvio Padrão F1 measure: {}'.format(stdev['f1m']))
    print('Desvio Padrão Acurácia: {}'.format(stdev['acc']))
    print('Desvio Padrão Sensibilidade (recall): {}'.format(stdev['rec']))
    print('Desvio Padrão Precisão: {}'.format(stdev['prec']))
    print('Desvio Padrão Especificidade: {}'.format(stdev['spec']))
    
