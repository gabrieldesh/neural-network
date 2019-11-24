import cross_validation as cv
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
        hits += cm[key]['vn']

    results = {
        'acc': 0,
        'rec': 0,
        'prec': 0,
        'spec': 0
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
        print('Matriz de confusão: VP {}, FP {}, VN {}, FN {}'.format(vp,fp,vn,fn))
        results['acc'] += accuracy
        results['rec'] += recall
        results['prec'] += precision
        results['spec'] += specificity
    print('\nResultados:')
    num_outputs = len(confusion_matrices)    
    results['acc'] = results['acc'] / num_outputs
    results['rec'] = results['rec'] / num_outputs
    results['prec'] = results['prec'] / num_outputs
    results['spec'] = results['spec'] / num_outputs
    if float(results['prec'] + results['rec']) > 0.0:
        results['f1m'] = float(2 * float(results['prec'] * results['rec']) / float(results['prec'] + results['rec']))
    else:
        results['f1m'] = 0.0
    print('\nF1 measure: {}'.format(results['f1m']))
    print('Acurácia: {}'.format(results['acc']))
    print('Sensibilidade (recall): {}'.format(results['rec']))
    print('Precisão: {}'.format(results['prec']))
    print('Especificidade: {}'.format(results['spec']))
