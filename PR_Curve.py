import numpy as np

def precision_recall(prob, label, threshold):
    res = np.zeros(prob.shape[0])
    precision = 0
    recall = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    
    for i in range(prob.shape[0]):
        if prob[i] >= threshold:
            res[i] = 1
            
    for i in range(prob.shape[0]):
        if res[i] == 1:
            if label[i] == 1: true_positive += 1
            elif label[i] == 0: false_positive += 1
            
        if res[i] == 0 and label[i] == 1:
            false_negative += 1
            
    if true_positive != 0:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
    else: return 0,0
    return precision, recall


def precision_recall_curve(predict_prob, label, threshold_list=None):
    recall_list = []
    precision_list = []
    
    if not threshold_list:
        threshold_list = [i/113 for i in range(113)]
        
    for threshold in threshold_list:
        P, R = precision_recall(predict_prob, label, threshold)
        recall_list.append(R)
        precision_list.append(P)
        
    return recall_list, precision_list, threshold_list

def F1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)
    
