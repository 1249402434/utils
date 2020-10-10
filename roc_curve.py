import numpy as np

def ROC(prob, label, threshold):
    res = np.zeros(prob.shape[0])
    TPR = 0
    FPR = 0
    true_positive = 0
    positive = np.sum(label == 1)
    false_positive = 0
    negative = np.sum(label == 0)
    
    for i in range(prob.shape[0]):
        if prob[i] >= threshold:
            res[i] = 1
            
    for i in range(prob.shape[0]):
        if res[i] == 1:
            if label[i] == 1: true_positive += 1
            elif label[i] == 0: false_positive += 1

    '''
    避免出现0作除数
    '''
    if positive == 0: TPR = 0
    else: TPR = true_positive / positive
        
    if negative == 0: FPR = 0
    else: FPR = false_positive / negative
        
    return TPR, FPR

def ROC_Curve(predict_prob, label, threshold_list=None):
    TPR_list = []
    FPR_list = []
    if not threshold_list: 
        threshold_list = [i/113 for i in range(113)]
        
    for threshold in threshold_list:
        TPR, FPR = ROC(predict_prob, label, threshold)
        TPR_list.append(TPR)
        FPR_list.append(FPR)
        
    return TPR_list, FPR_list
