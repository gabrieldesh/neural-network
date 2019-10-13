import cross_validation as cv
from train import train_neural_network
from predict import predict_all

def eval_confusion_matrices(dataset, network_structure, initial_weights, 
                            learning_rate, momentum, batch_size, k):

    cvset = cv.cross_validation(dataset,k)
    
    results = {}
    for i in range(k):
        target = cvset[i]['testSet']['target']
        print('Generating neural net for the k-fold {}'.format(i))
        neural_network = train_neural_network(cvset[i]['trainingSet'], network_structure, initial_weights, 
                                              learning_rate, momentum, batch_size)
        predictions = predict_all(neural_network, cvset[i]['testSet'])
        result = {}
        vals = []
        for val in cvset[i]['testSet']['attributes'][target]['values']:
            vals.append(val)
            counts = {}
            for val2 in cvset[i]['testSet']['attributes'][target]['values']:
                counts[val2] = 0
            # a linha abaixo conserta um bug
            cvset[i]['testSet']['data'][target] = cvset[i]['testSet']['data'][target].astype('category')
            for j in range(len(cvset[i]['testSet']['data'])):
                if str(cvset[i]['testSet']['data'].iloc[j][target]) == str(val):
                    counts[str(predictions[j])] = counts[str(predictions[j])] + 1
            result[val] = counts
        results[i] = result
    
    return {
        'vals' : vals,
        'results': results
    }

def summarize_matrices(confusion_matrices):

    cmatrix = {}
    for i in range(len(confusion_matrices['vals'])):
        #print('\nClasse considerada positiva: {}'.format(confusion_matrices['vals'][i]))
        vp = 0
        fp = 0
        vn = 0
        fn = 0
        for j in range(len(confusion_matrices['results'])):
            #print(confusion_matrices['results'][j])
            for k in range(len(confusion_matrices['vals'])):
                #print(confusion_matrices['vals'][k],confusion_matrices['results'][j][confusion_matrices['vals'][k]])
                for l in range(len(confusion_matrices['vals'])):
                    #print(confusion_matrices['results'][j][confusion_matrices['vals'][k]][confusion_matrices['vals'][l]])
                    #print(confusion_matrices['vals'][i] == confusion_matrices['vals'][k])
                    #print(confusion_matrices['vals'][i] == confusion_matrices['vals'][l])
                    # verdadeiro positivo
                    if confusion_matrices['vals'][i] == confusion_matrices['vals'][k] and confusion_matrices['vals'][i] == confusion_matrices['vals'][l]:
                        vp = vp + confusion_matrices['results'][j][confusion_matrices['vals'][k]][confusion_matrices['vals'][l]]
                    # falso negativo
                    if confusion_matrices['vals'][i] == confusion_matrices['vals'][k] and not confusion_matrices['vals'][i] == confusion_matrices['vals'][l]:
                        fn = fn + confusion_matrices['results'][j][confusion_matrices['vals'][k]][confusion_matrices['vals'][l]]
                    # falso positivo
                    if not confusion_matrices['vals'][i] == confusion_matrices['vals'][k] and confusion_matrices['vals'][i] == confusion_matrices['vals'][l]:
                        fp = fp + confusion_matrices['results'][j][confusion_matrices['vals'][k]][confusion_matrices['vals'][l]]
                    # verdadeiro negativo
                    if not confusion_matrices['vals'][i] == confusion_matrices['vals'][k] and not confusion_matrices['vals'][i] == confusion_matrices['vals'][l]:
                        vn = vn + confusion_matrices['results'][j][confusion_matrices['vals'][k]][confusion_matrices['vals'][l]]
        summary = { 'vp' : vp, 'fn': fn, 'fp': fp, 'vn': vn, 'summ': vp+fn+fp+vn}
        cmatrix[confusion_matrices['vals'][i]] = summary

    return cmatrix

def eval_f1measure(confusion_matrices, positive_class = None):

    cm = summarize_matrices(confusion_matrices)
    hits = 0

    for key in cm:
        hits += cm[key]['vp']

    for key in cm:

        if positive_class == None or positive_class == key:

            print('\nResumo estatístico considerando {} como classe positiva:'.format(key))
            vp = cm[key]['vp']
            fp = cm[key]['fp']
            vn = cm[key]['vn']
            fn = cm[key]['fn']
            summ = cm[key]['summ']
            accuracy = float(hits) / float(summ)
            recall = float(vp) / float(vp + fn)
            precision = float(vp) / float(vp+fp)
            specificity = float(vn) / float(vn+fp)
            f1m = float(2 * float(precision * recall) / float(precision + recall))
            print('\nF1 measure: {}'.format(f1m))
            print('Acurácia: {}'.format(accuracy))
            print('Sensibilidade (recall): {}'.format(recall))
            print('Precisão: {}'.format(precision))
            print('Especificidade: {}'.format(specificity))
            print('Matriz de confusão: VP {}, FP {}, VN {}, FN {}'.format(vp,fp,vn,fn))
