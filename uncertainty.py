import numpy as np


def softmax(pred):
    ex = np.exp(pred - np.amax(pred, axis=1, keepdims=True))
    prob = ex / np.sum(ex, axis=1, keepdims=True)
    return prob


def entropy(mean):
    class_num = mean.shape[1]
    prob = softmax(mean)
    entropy = - prob * (np.log(prob + 5e-10) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un


def total_uncertainty(Baye_result):
    prob_all = []
    class_num = Baye_result[0].shape[1]
    for item in Baye_result:
        prob = softmax(item)
        prob_all.append(prob)
    prob_mean = np.mean(prob_all, axis=0)
    total_class_un = - prob_mean * (np.log(prob_mean + 5e-10) / np.log(class_num))
    total_un = np.sum(total_class_un, axis=1, keepdims=True)
    return total_un, total_class_un


def aleatoric_uncertainty(Baye_result):
    al_un = []
    al_class_un = []
    for item in Baye_result:
        un, class_un = entropy(item)
        al_un.append(un)
        al_class_un.append(class_un)
    ale_un = np.mean(al_un, axis=0)
    ale_class_un = np.mean(al_class_un, axis=0)
    return ale_un, ale_class_un


def get_uncertainty(Baye_result):
    uncertainty = []
    uncertainty_class = []
    un_total, un_total_class = total_uncertainty(Baye_result)
    un_aleatoric, un_aleatoric_class = aleatoric_uncertainty(Baye_result)
    un_epistemic_class = un_total_class - un_aleatoric_class
    un_epistemic = np.sum(un_epistemic_class, axis=1, keepdims=True)
    uncertainty.append(un_aleatoric)
    uncertainty.append(un_epistemic)
    uncertainty.append(un_total)

    uncertainty_class.append(un_aleatoric_class)
    uncertainty_class.append(un_epistemic_class)
    uncertainty_class.append(un_total_class)

    return uncertainty

def get_epistemic_uncertainty(Baye_result):
    return get_uncertainty(Baye_result)[1]
