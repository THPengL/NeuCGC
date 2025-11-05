
import numpy as np
from munkres import Munkres
import sklearn.metrics as metrics
from sklearn.metrics import adjusted_rand_score, accuracy_score, f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score 



def clustering_acc(y_true, y_pred):
    """
    Calculate clustering accuracy and f1-score.

    Parameters
    - y_true: the ground truth.
    - y_pred: the predicted clustering ids.

    Returns
    - acc: clustering accuracy.
    - f1-score: macro f1-score of clustering result.
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = accuracy_score(y_true, new_predict)
    f1_macro = f1_score(y_true, new_predict, average='macro')

    return acc, f1_macro


def evaluate(y_true, y_pred):
    """
    Evaluate the clustering performance.

    Parameters
    - y_true: The ground truth
    - y_pred: The predicted label

    Return
    - acc: Clustering accuracy.
    - nmi: Normalized mutual information.
    - ari: Adjusted rand score.
    - f1: F1-score.
    """
    acc, f1 = clustering_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
        
    return acc, nmi, ari, f1

