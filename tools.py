# -*- coding: utf-8 -*-


from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import ndimage
from sklearn import decomposition
import scipy as sp;
import pandas as pd;
import os;
import glob;
import numpy as np;
from sklearn import svm;

import heapq; ##to get n largest num from a list
import matplotlib.cm as cm;##import a color map
import sklearn.metrics.pairwise as measure;
import numpy as np;
from sklearn.preprocessing import normalize;
from scipy.optimize import minimize;
import multiprocessing as mp
from sklearn.gaussian_process.kernels import RBF;
import time;

#generate copies of train,test and candidate index for different AL
def copyIndex(trainInd,testInd,poolInd):
    return [x for x in trainInd],[y for y in testInd],[z for z in poolInd];


def smothCurve(line,fraction=0.20):
    filtered = lowess(line, range(len(line)), is_sorted=True, frac=fraction, it=0);
    x=range(len(filtered[:,0]))
    y=filtered[:,1]
    return x,y;

def testPartition(Y,train_n_candidate=0.66,test=0.33,ntrain=3):
    allIndex=range(len(Y))
    np.random.shuffle(allIndex)
    testIndex=allIndex[0:int(len(allIndex)*test)]
    trainIndex=allIndex[int(len(allIndex)*test):int(len(allIndex)*test)+ntrain]
    poolIndex=allIndex[int(len(allIndex)*test)+ntrain:]       
    return trainIndex,testIndex,poolIndex
        
#return the index of train, test and pool. make sure at least leastTrain instances are in the training for each class. at least leastPool instances are in the pool. the rest will be in the test
def partitionForAL(Y,leastTrain=1,leastPool=30,randomSeed=0,signi=None,NumClass=None,designatedClass=None):
    trainIndex=[]
    testIndex=[]
    poolIndex=[]
    classLen=0
    #decide how many class to use
    if (NumClass is None):
        classLen=len(list(set(Y)))
    else:
        classLen=NumClass
    if(designatedClass is None):
        classList=list(set(Y))
    else:
        classList=designatedClass
    np.random.seed(randomSeed)
    np.random.shuffle(classList)        
    #if class distribution is balanced, can directly assign how many data instance per class for train/candidate
    if(leastTrain>=1 or leastPool>=1):
        for className in list(set(Y)):
            #get the index of each class
            classIndex=[i for i in range(len(Y)) if Y[i]==className]
            if signi is None:
                np.random.seed(randomSeed)                    
                np.random.shuffle(classIndex)
            else:#if pass a significance list of samples,(how many none zero emptys in the data.)    
                classIndex=[classIndex[k] for k in np.argsort(signi[classIndex])]
                #classIndex[np.argsort(signi[classIndex])]
            if (className in classList[0:classLen]):
                trainIndex=trainIndex+classIndex[0:leastTrain]
            poolIndex=poolIndex+classIndex[leastTrain:leastTrain+leastPool]
            testIndex=testIndex+classIndex[leastTrain+leastPool:len(classIndex)]
        return trainIndex,testIndex,poolIndex
    #for unbalanced class distribution, assign the percentage of instances for each class used for train/candidate    
    if(leastTrain<1 or leastPool<1):
        for className in list(set(Y)):
            #get the index of each class
            classIndex=[i for i in range(len(Y)) if Y[i]==className]    
            np.random.seed(randomSeed)                    
            np.random.shuffle(classIndex)
            trainIndex=trainIndex+classIndex[0:int(np.ceil(leastTrain*len(classIndex)))]
            poolIndex=poolIndex+classIndex[int(np.ceil(leastTrain*len(classIndex))):int(np.ceil((leastTrain+leastPool)*len(classIndex)))]                                             
            testIndex=testIndex+classIndex[int(np.ceil((leastTrain+leastPool)*len(classIndex))):len(classIndex)]                                
        return trainIndex,testIndex,poolIndex
        
        
        

    