# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
from scipy.spatial import ConvexHull
from scipy.optimize import minimize;
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as nd
import scipy as sp
import pylab as pl
from IPython import display
from tensorflow.examples.tutorials.mnist import input_data
print("GPU Available: ", tf.test.is_gpu_available())
import sklearn as sk
from sklearn.gaussian_process.kernels import RBF;
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# define some utility functions
import pickle
import os
import tools
os.chdir('D:\\projects\\uncertainty') 
from uncertainty import get_epistemic_uncertainty
from convexHullCheck import *
import importlib


##############utils

def var(name, shape, init=None):
    if init is None:
        init = tf.truncated_normal_initializer(stddev=(2 / shape[0]) ** 0.5)
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                           initializer=init)

def conv(Xin, f, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(Xin, f, strides, padding)

def max_pool(Xin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(Xin, ksize, strides, padding)


def LeNetSoftMax_dropout(lmb=0.005):
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(shape=[None, 28 * 28], dtype=tf.float32)
        Y = tf.placeholder(shape=[None, 10], dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32)
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        annealing_step = tf.placeholder(dtype=tf.int32)
        # first hidden layer - conv
        W1 = var('W1', [5, 5, 1, 20])
        b1 = var('b1', [20])
        out1 = tf.nn.relu(conv(tf.reshape(X, [-1, 28, 28, 1]), W1, strides=[1, 1, 1, 1]) + b1)
        out1 = tf.nn.dropout(out1, keep_prob=keep_prob)
        out1 = max_pool(out1)
        # second hidden layer - conv
        W2 = var('W2', [5, 5, 20, 50])
        b2 = var('b2', [50])
        out2 = tf.nn.relu(conv(out1, W2, strides=[1, 1, 1, 1]) + b2)
        out2 = tf.nn.dropout(out2, keep_prob=keep_prob)
        out2 = max_pool(out2)
        # flatten the output
        Xflat = tf.contrib.layers.flatten(out2)
        # third hidden layer - fully connected
        W3 = var('W3', [Xflat.get_shape()[1].value, 500])
        b3 = var('b3', [500])
        out3 = tf.nn.relu(tf.matmul(Xflat, W3) + b3)
        out3 = tf.nn.dropout(out3, keep_prob=keep_prob)
        # output layer
        W4 = var('W4', [500, 10])
        b4 = var('b4', [10])
        logits = tf.matmul(out3, W4) + b4
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        l2_loss = (tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4))
        step = tf.train.AdamOptimizer().minimize(loss + l2_loss * lmb, global_step=global_step)
        return g, step, X, Y, annealing_step, keep_prob, loss, logits


def accuracy_array(y_predit, test_y):
    y_predit = np.argmax(y_predit, axis=1)
    test_y = np.argmax(test_y, axis=1)
    correct_prediction = np.equal(y_predit, test_y)
    acc = np.mean(correct_prediction)
    return acc


def load_minist(X,Y,trainInd,candInd,testInd):
    '''
    '''
    return X, Y, trainInd, candInd, testInd

#BALD method
def train(X,Y,trainInd,candInd,testInd,AL_iter=300, epoch=200, dropout_inf=20, sample_num=1):
    # load data
    data, label, trainInd, candInd, testInd = load_minist(X,Y,trainInd,candInd,testInd)
    acc_result = []
    # load model
    g, step, X, Y, annealing_step, keep_prob, loss, logits = LeNetSoftMax_dropout()
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    sess = tf.Session(graph=g)#, config=config)
    with g.as_default():
        for i in range(AL_iter):
            print('AL iter:'+str(AL_iter))
            ## initialize weight
            sess.run(tf.global_variables_initializer())
            train_data = data[trainInd, :]
            train_label = label[trainInd, :]
            test_data = data[testInd, :]
            test_label = label[testInd, :]
            cand_data = data[candInd, :]
            cand_label = label[candInd, :]
            for e in range(epoch):
                sess.run(step, {X: train_data, Y: train_label, keep_prob: .5})
            Bayesian_result_test = []
            Bayesian_result_cand = []
            for j in range(dropout_inf):
                output_test = sess.run(logits, feed_dict={X: test_data, Y: test_label, keep_prob: .5})
                Bayesian_result_test.append(output_test)
                output_cand = sess.run(logits, feed_dict={X: cand_data, Y: cand_label, keep_prob: .5})
                Bayesian_result_cand.append(output_cand)
            test_acc = accuracy_array(np.mean(Bayesian_result_test, axis=0), test_label)

            ## sample
            epistemic_uncertainty = get_epistemic_uncertainty(Bayesian_result_cand)
            sortInd = np.argsort(epistemic_uncertainty, axis=0)[::-1]
            sortInd = [int(x) for x in sortInd]
            temInd = [candInd[sortInd[x]] for x in sortInd[0:sample_num]]
            trainInd = trainInd + temInd
            candInd = [x for x in candInd if x not in temInd]
            print('Al_iter: ', i, ' testing accuracy: %2.4f' % (test_acc))
            acc_result.append(test_acc)
    np.save('d:\\projects\\uncertainty\\Bayesian_minist_acc.npy', acc_result)
    return acc_result

##################################################################################
def var(name, shape, init=None):
    if init is None:
        init = tf.truncated_normal_initializer(stddev=(2/shape[0])**0.5)
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                          initializer=init)

def conv(Xin, f, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(Xin, f, strides, padding)

def max_pool(Xin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(Xin, ksize, strides, padding)

def rotate_img(x, deg):
    import scipy.ndimage as nd
    return nd.rotate(x.reshape(28,28),deg,reshape=False).ravel()
#evidential dnn
# This one usually works better and used for the second and third examples
# For general settings and different datasets, you may try this one first
def exp_evidence(logits): 
    return tf.exp(tf.clip_by_value(logits,-10,10))

#generate simple synthetic data
def syntheticDataGen(means=None,covs=None,samplePerClass=500,delta=4):#generate Gaussian synthetic data
    #like in prior nn, use delta to control overlap
    istroVar=.1*delta*delta
    if(means==None):
        a1=np.random.multivariate_normal([2.5,2.5], [[istroVar,0],[0,istroVar]], samplePerClass)
        a2=np.random.multivariate_normal([.5,0.5], [[istroVar,0],[0,istroVar]], samplePerClass)
        a3=np.random.multivariate_normal([4.5,0.5], [[istroVar,0],[0,istroVar]], samplePerClass)
                #plt.plot(x, y, 'x')
        x=np.concatenate([a1,a2,a3],axis=0)
        #Create pdfs
        n1=multivariate_normal(mean=[2.5,2.5], cov=[[istroVar,0],[0,istroVar]])
        n2=multivariate_normal(mean=[.5,.5], cov=[[istroVar,0],[0,istroVar]])
        n3=multivariate_normal(mean=[4.5,0.5], cov=[[istroVar,0],[0,istroVar]])
        y=np.zeros([samplePerClass*3,3])
        # true class distri of the samples(not normalized)
        trueProb=np.zeros([samplePerClass*3,3])
        for i in range(3):
            y[i*samplePerClass:(i+1)*samplePerClass,i]=1
            trueProb[:,i]=[n1,n2,n3][i].pdf(x)
         # true class distri entropy of the samples(not normalized)            
        trueEn=sp.stats.entropy(trueProb.T)
        return x,y,trueProb,trueEn,[n1,n2,n3]
    else:
        return None
    
    #generate complex synthetic data (GMM)
def syntheticDataGen2(means=None,covs=None,samplePerClass=500,delta=2,outlierSize=50,outlierDelta=1):#generate Mix Gaussian synthetic data 
    #like in prior nn, use delta to control overlap
    istroVar=.1*delta*delta
    istroVar2=.1*outlierDelta*outlierDelta
    #class weight in gmm
    largeWeight=samplePerClass/(samplePerClass*3+outlierSize*3)
    smallWeight=outlierSize/(samplePerClass*3+outlierSize*3)
    if(means==None):
        a1=np.random.multivariate_normal([2.5,2.5], [[istroVar,0],[0,istroVar]], samplePerClass)
        a2=np.random.multivariate_normal([.5,0.5], [[istroVar,0],[0,istroVar]], samplePerClass)
        a3=np.random.multivariate_normal([4.5,0.5], [[istroVar,0],[0,istroVar]], samplePerClass)
        b3=np.random.multivariate_normal([-2,8.5], [[istroVar2,0],[0,istroVar2]],outlierSize )        
        b2=np.random.multivariate_normal([8,8.5], [[istroVar2,0],[0,istroVar2]],outlierSize )        
        b1=np.random.multivariate_normal([2.5,-6.5], [[istroVar2,0],[0,istroVar2]],outlierSize )        
        #plt.plot(x[:,0], x[:,1], 'x')
        x=np.concatenate([a1,b1,a2,b2,a3,b3],axis=0)
        #Create pdfs
        n1=multivariate_normal(mean=[2.5,2.5], cov=[[istroVar,0],[0,istroVar]])
        n2=multivariate_normal(mean=[.5,.5], cov=[[istroVar,0],[0,istroVar]])
        n3=multivariate_normal(mean=[4.5,0.5], cov=[[istroVar,0],[0,istroVar]])
        nn3=multivariate_normal([-2,8.5], [[istroVar2,0],[0,istroVar2]])
        nn2=multivariate_normal([8,8.5], [[istroVar2,0],[0,istroVar2]])
        nn1=multivariate_normal([2.5,-6.5], [[istroVar2,0],[0,istroVar2]])
        
        y=np.zeros([(samplePerClass+outlierSize)*3,3])
        # true class distri of the samples(not normalized)
        trueProb=np.zeros([(samplePerClass+outlierSize)*3,3])#we wont need this
        for i in range(3):
            y[i*(samplePerClass+outlierSize):(i+1)*(samplePerClass+outlierSize),i]=1
            #trueProb[:,i]=[n1,n2,n3][i].pdf(x)
         # true class distri entropy of the samples(not normalized)            
        trueEn=sp.stats.entropy(trueProb.T)#we wont need this
        #plot true post here once for all
        n_grid = 500
        max_x      = np.max(x,axis = 0)
        min_x      = np.min(x,axis = 0)
        XX1         = np.linspace(min_x[0],max_x[0],n_grid)
        XX2         = np.linspace(min_x[1],max_x[1],n_grid)
        x1,x2      = np.meshgrid(XX1,XX2)
        Xgrid      = np.zeros([n_grid**2,2])
        Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
        Xgrid[:,1] = np.reshape(x2,(n_grid**2,))
        trueProb=np.zeros([Xgrid.shape[0],3])
        pdfs=[[n1,nn1],[n2,nn2],[n3,nn3]]
        for i in range(3):
            trueProb[:,i]=pdfs[i][0].pdf(Xgrid)*largeWeight+pdfs[i][1].pdf(Xgrid)*smallWeight
        #get the entropy of each grid            
        trueEn=sp.stats.entropy(trueProb.T)  
        a=plt.figure(figsize=(25,16))
        plt.contourf(XX1,XX2,np.reshape(trueEn,(n_grid,n_grid)),cmap="coolwarm",figsize = (20,12))
        plt.colorbar()
        ydim=np.argmax(y,axis=1)
        plt.plot(x[ydim==0,0],x[ydim==0,1],"ro", markersize = 3)
        plt.plot(x[ydim==1,0],x[ydim==1,1],"ks", markersize = 3)
        plt.plot(x[ydim==2,0],x[ydim==2,1],"b^", markersize = 3)
        return x,y,trueProb,trueEn,pdfs
    else:
        return None    
#x,y,trueP,trueE,pdfs=syntheticDataGen(delta=4)
def OODIdentify(X,trainIndex,candidateIndex,type='trainNCandidate',top=0.05,top2=0.5):
    if (type=='trainNCandidate'):
    #if we treate train and candidate as a whole, we identify the farthest point.
    #compute data center, list farthest points within training.
    #top:% of data points need to be considered as OOD
        data=X[trainIndex+candidateIndex,:]
        center=np.mean(data,axis=0);
        center=center[np.newaxis,:]
        rbf=RBF();
        kernelDist=rbf.__call__(data,center)
        OODIndex=list(np.argsort(kernelDist,axis=0)[:,0])[0:int(np.ceil(X.shape[0]*top))]
        return OODIndex #this is the index of data(x_train+x_candidate)
    if (type=='trainVsCandidate'):
        #in this case we use cvx hull of train to identify OOD in candidate(In other words we assume all current training are in distribution data.)
        def pnt2line(pnt, start, end):
            line_vec = vector(start, end)
            pnt_vec = vector(start, pnt)
            line_len = length(line_vec)
            line_unitvec = unit(line_vec)
            pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
            t = dot(line_unitvec, pnt_vec_scaled)    
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            nearest = scale(line_vec, t)
            dist = distance(nearest, pnt_vec)
            nearest = add(nearest, start)
            return (dist, nearest)
        
        data=X[trainIndex,:]
        hull = ConvexHull(data)
        pt_dist = []
        for p_idx in candidateIndex:
            pt = X[p_idx,:]
            dist_list = []
            for v_idx in range(len(hull.vertices)):
                v1 = hull.vertices[v_idx - 1]
                v2 = hull.vertices[v_idx]
                start = data[v1]
                end = data[v2]
                temp = pnt2line(pt, start, end)
                dist_list.append(temp[0])
        #check points inside or out the polygon
            inside =  point_in_poly(pt[0],pt[1],data[hull.vertices])
            if (inside == True):
                dist_temp = -1. * min(dist_list)
            else:
                dist_temp = min(dist_list)                  
            pt_dist.append(dist_temp)
            #identify half of the training size of candidates as OOD
        OODIndex = list(len(trainIndex)+np.argsort(pt_dist))[::-1][0:int(np.ceil(X.shape[0]*top))]#this is also the index of data(x_train+x_candidate)
    #the training data should be like train+candidate. the train data and its label will be used for noramal loss. and OODIndex of the data will be used for OOD loss(alpha)       
        return OODIndex#,pt_dist
    if (type=='minMax'):
        data=X[trainIndex,:]
        candi=X[candidateIndex,:]
        rbf=RBF();
        kernelDist=rbf.__call__(data,candi)
        #note min dist=max kernelDist
        OODIndex=list(len(trainIndex)+np.argsort(np.max(kernelDist,axis=0)))[0:int(np.ceil(len(candidateIndex)*top))]
        return OODIndex, np.min(np.max(kernelDist,axis=0))
    if  (type=='minMax_full'):
        data=X[trainIndex+candidateIndex,:]
        #candi=X[candidateIndex,:]
        rbf=RBF();
        kernelDist=rbf.__call__(data,data)
        np.fill_diagonal(kernelDist,np.zeros(kernelDist.shape[0]))
        #note min dist=max kernelDist
        OODIndex=list(np.argsort(np.max(kernelDist,axis=0)))[0:int(np.ceil(X.shape[0]*top))]        
        return OODIndex,None
    if (type=='minMax_double'):
        data=X[trainIndex,:]
        candi=X[candidateIndex,:]
        rbf=RBF();
        kernelDist=rbf.__call__(data,candi)
        #candidate ood index only need top% of them
        oodCandidateInd=np.argsort(np.max(kernelDist,axis=0))[0:int(np.ceil(len(candidateIndex)*top))]
        ood=candi[oodCandidateInd,:]
        #form the gram mat again
        kernelDist2=rbf.__call__(ood,ood)
        #this time choose half of them,do not forget to reset diag of gram since this is the self-self comparasion
        np.fill_diagonal(kernelDist2,np.zeros(kernelDist2.shape[0]))
        temInd=np.argsort(np.max(kernelDist2,axis=0))[0:int(np.ceil(len(oodCandidateInd)*top2))]
        #this is the index of oodCandidateInd, convert to index of candidate and convert again to train+candidate index
        OODIndex=list(len(trainIndex)+oodCandidateInd[temInd])
        return OODIndex, np.min(np.max(kernelDist2,axis=0))
    if (type=='kernelDensity'):
        data=X[trainIndex,:]
        candi=X[candidateIndex,:]
        rbf=RBF();
        kernelDist=rbf.__call__(data,candi)
        oodCandidateInd=np.argsort(np.sum(kernelDist,axis=0))[0:int(np.ceil(len(candidateIndex)*top))]
        ood=candi[oodCandidateInd,:]
        kernelDist2=rbf.__call__(ood,ood)
        np.fill_diagonal(kernelDist2,np.zeros(kernelDist2.shape[0]))
        temInd=np.argsort(np.sum(kernelDist2,axis=0))[::-1][0:int(np.ceil(len(oodCandidateInd)*top2))]
        OODIndex=list(len(trainIndex)+oodCandidateInd[temInd])
        return OODIndex,  np.min(np.max(kernelDist,axis=0))
    if (type=='kernelDensity2'):
        data=X[trainIndex,:]
        candi=X[candidateIndex,:]
        rbf=RBF();
        kernelDist=rbf.__call__(data,candi)
        kernelDist2=rbf.__call__(candi,candi)
        p1=np.sum(kernelDist,axis=0)/len(trainIndex)
        p2=np.sum(kernelDist2,axis=0)/len(candidateIndex)
        diff=top*p2-p1
        #who is positive?
        posDiff=np.where(diff>=0)[0]
        #sort the positive
        tem=np.argsort(diff[posDiff])[::-1]
        OODIndex=list(len(trainIndex)+posDiff[tem][0:15])
        return OODIndex,np.min(np.max(kernelDist,axis=0))
        
    #
def testOODIdentify(X,trainIndex,candidateIndex,top=0.02,top2=0.5):
    data=X[trainIndex,:]
    candi=X[candidateIndex,:]
    rbf=RBF();
    kernelDist=rbf.__call__(data,candi)
    kernelDist2=rbf.__call__(candi,candi)
    p1=np.sum(kernelDist,axis=0)/len(trainIndex)
    p2=np.sum(kernelDist2,axis=0)/len(candidateIndex)
    diff=top*p2-p1
    #who is positive?
    posDiff=np.where(diff>=0)[0]
    #sort the positive
    tem=np.argsort(diff[posDiff])[::-1]
    OODIndex=list(len(trainIndex)+posDiff[tem][0:30])
    return OODIndex,np.min(np.max(kernelDist,axis=0)),p1,p2

def dissonance(alpha):
    #def Bal(i,j)
    def Bal(b,i,j):
        if (b[i]+b[j] ==0):
            return 0
        else:
            return 1-np.abs(b[i]-b[j])/(b[i]+b[j])
    #get evidence
    evi=alpha-1
    #get b
    b=evi/np.sum(alpha)
    res=0
    for i in range(len(b)):
        #print('i'+str(i))
        excludeInd=[x for x in range(len(b)) if x != i]
        tem1=0
        tem2=0
        for j in excludeInd:
            #print('j'+str(j))
            tem1+=(b[j]*Bal(b,i,j))
            tem2+=(b[j])
        if(tem2==0):
            #print(0)
            return 0
        res+=b[i]*tem1/tem2
    #print(res)
    return res            

    

def KL(alpha):
    beta=tf.constant(np.ones((1,K)),dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha,axis=1,keep_dims=True)
    S_beta = tf.reduce_sum(beta,axis=1,keep_dims=True)
    lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha),axis=1,keep_dims=True)
    lnB_uni = tf.reduce_sum(tf.lgamma(beta),axis=1,keep_dims=True) - tf.lgamma(S_beta)
    
    dg0 = tf.digamma(S_alpha)
    dg1 = tf.digamma(alpha)
    
    kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keep_dims=True) + lnB + lnB_uni
    return kl

def mse_loss(p, alpha, global_step, annealing_step): 
    S = tf.reduce_sum(alpha, axis=1, keep_dims=True) 
    E = alpha - 1
    m = alpha / S
    
    A = tf.reduce_sum((p-m)**2, axis=1, keep_dims=True) 
    B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keep_dims=True) 
    
    annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
    
    alp = E*(1-p) + 1 
    C =  annealing_coef * KL(alp)
    return (A + B) + C

def relu_evidence(logits):
    return tf.nn.relu(logits)

def mine2(X,reuse=False,keep_prob=None):
    with tf.variable_scope("mine2", reuse=reuse) as scope:
        W1 = var('W1', [5,5,1,20])
        b1 = var('b1', [20])
        out1 = max_pool(tf.nn.relu(conv(tf.reshape(X, [-1, 28,28, 1]), 
                                        W1, strides=[1, 1, 1, 1]) + b1))
        # second hidden layer - conv
        W2 = var('W2', [5,5,20,50])
        b2 = var('b2', [50])
        out2 = max_pool(tf.nn.relu(conv(out1, W2, strides=[1, 1, 1, 1]) + b2))
        # flatten the output
        Xflat = tf.contrib.layers.flatten(out2)
        # third hidden layer - fully connected
        W3 = var('W3', [Xflat.get_shape()[1].value, 500])
        b3 = var('b3', [500]) 
        out3 = tf.nn.relu(tf.matmul(Xflat, W3) + b3)
        out3 = tf.nn.dropout(out3, keep_prob=keep_prob)
        #output layer
        W4 = var('W4', [500,10])
        b4 = var('b4',[10])
        logits = tf.matmul(out3, W4) + b4
        return logits,tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)

def LeNet_Reg_EDL(logits2evidence=relu_evidence,loss_function=mse_loss, reguScale=0.005, lmb=0.005,activation=None,network=mine2):
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(shape=[None,28*28], dtype=tf.float32)
        Y = tf.placeholder(shape=[None,10], dtype=tf.float32)
        XX = tf.placeholder(shape=[None,28*28], dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32)
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        annealing_step = tf.placeholder(dtype=tf.int32) 
        logits,l2_loss = network(X,reuse=False,keep_prob=keep_prob)
        oodLogits,ood_l2_loss=network(XX,reuse=True,keep_prob=keep_prob)
        
        
        ooD_evidence = logits2evidence(oodLogits)
        ooD_alpha = ooD_evidence + 1
        
        evidence = logits2evidence(logits)
        alpha = evidence + 1
        
        #u = K / tf.reduce_sum(alpha, axis=1, keep_dims=True) #uncertainty
        u = K / tf.reduce_sum(ooD_alpha, axis=1, keep_dims=True) #uncertainty
        oodLoss=tf.reduce_sum(u)
        prob = alpha/tf.reduce_sum(alpha, 1, keepdims=True) 
        
        loss = tf.reduce_mean(loss_function(Y, alpha, global_step, annealing_step))
        step = tf.train.AdamOptimizer().minimize(loss+l2_loss * reguScale - oodLoss*lmb , global_step=global_step)
        
        # Calculate accuracy
        pred = tf.argmax(logits, 1)
        truth = tf.argmax(Y, 1)
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
        acc = tf.reduce_mean(match)
        
        total_evidence = tf.reduce_sum(evidence,1, keepdims=True) 
        mean_ev = tf.reduce_mean(total_evidence)
        mean_ev_succ = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*match) / tf.reduce_sum(match+1e-20)
        mean_ev_fail = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
        return g, step, X, Y,XX, annealing_step, keep_prob, prob, acc, loss, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail,alpha,total_evidence, logits,truth,pred

def LeNetSoftMax(logits2evidence=relu_evidence,loss_function=mse_loss, lmb=0.005,activation=None):
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(shape=[None,28*28], dtype=tf.float32)
        Y = tf.placeholder(shape=[None,10], dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32)
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        annealing_step = tf.placeholder(dtype=tf.int32) 
        # first hidden layer - conv
        W1 = var('W1', [5,5,1,20])
        b1 = var('b1', [20])
        out1 = max_pool(tf.nn.relu(conv(tf.reshape(X, [-1, 28,28, 1]), 
                                        W1, strides=[1, 1, 1, 1]) + b1))
        # second hidden layer - conv
        W2 = var('W2', [5,5,20,50])
        b2 = var('b2', [50])
        out2 = max_pool(tf.nn.relu(conv(out1, W2, strides=[1, 1, 1, 1]) + b2))
        # flatten the output
        Xflat = tf.contrib.layers.flatten(out2)
        # third hidden layer - fully connected
        W3 = var('W3', [Xflat.get_shape()[1].value, 500])
        b3 = var('b3', [500]) 
        out3 = tf.nn.relu(tf.matmul(Xflat, W3) + b3)
        out3 = tf.nn.dropout(out3, keep_prob=keep_prob)
        #output layer
        W4 = var('W4', [500,10])
        b4 = var('b4',[10])
        logits = tf.matmul(out3, W4) + b4
        prob = tf.nn.softmax(logits=logits) 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        l2_loss = (tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4))
        step = tf.train.AdamOptimizer().minimize(loss+l2_loss * lmb, global_step=global_step)
        # Calculate accuracy
        pred = tf.argmax(logits, 1)
        truth = tf.argmax(Y, 1)
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
        acc = tf.reduce_mean(match)
        alpha= tf.zeros([3, 4], tf.int32)
        return g, step, X, Y,annealing_step, keep_prob, prob, acc, loss,None, None, None, None, None,alpha,None, logits,truth,pred

def LeNet_EDL(logits2evidence=relu_evidence,loss_function=mse_loss, lmb=0.005,activation=None):
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(shape=[None,28*28], dtype=tf.float32)
        Y = tf.placeholder(shape=[None,10], dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32)
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        annealing_step = tf.placeholder(dtype=tf.int32) 
    
        # first hidden layer - conv
        W1 = var('W1', [5,5,1,20])
        b1 = var('b1', [20])
        out1 = max_pool(tf.nn.relu(conv(tf.reshape(X, [-1, 28,28, 1]), 
                                        W1, strides=[1, 1, 1, 1]) + b1))
        # second hidden layer - conv
        W2 = var('W2', [5,5,20,50])
        b2 = var('b2', [50])
        out2 = max_pool(tf.nn.relu(conv(out1, W2, strides=[1, 1, 1, 1]) + b2))
        # flatten the output
        Xflat = tf.contrib.layers.flatten(out2)
        # third hidden layer - fully connected
        W3 = var('W3', [Xflat.get_shape()[1].value, 500])
        b3 = var('b3', [500]) 
        out3 = tf.nn.relu(tf.matmul(Xflat, W3) + b3)
        out3 = tf.nn.dropout(out3, keep_prob=keep_prob)
        #output layer
        W4 = var('W4', [500,10])
        b4 = var('b4',[10])
        logits = tf.matmul(out3, W4) + b4
        evidence = logits2evidence(logits)
        alpha = evidence + 1
        u = K / tf.reduce_sum(alpha, axis=1, keep_dims=True) #uncertainty
        prob = alpha/tf.reduce_sum(alpha, 1, keepdims=True) 
        loss = tf.reduce_mean(loss_function(Y, alpha, global_step, annealing_step))
        l2_loss = (tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4))
        step = tf.train.AdamOptimizer().minimize(loss+l2_loss * lmb, global_step=global_step)
        # Calculate accuracy
        pred = tf.argmax(logits, 1)
        truth = tf.argmax(Y, 1)
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
        acc = tf.reduce_mean(match)
        total_evidence = tf.reduce_sum(evidence,1, keepdims=True) 
        mean_ev = tf.reduce_mean(total_evidence)
        mean_ev_succ = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*match) / tf.reduce_sum(match+1e-20)
        mean_ev_fail = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
        return g, step, X, Y, annealing_step, keep_prob, prob, acc, loss, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail,alpha,total_evidence, logits,truth,pred

def simple_EDL(logits2evidence=relu_evidence,loss_function=mse_loss, lmb=0.005,activation='linear',reguScale=0.0005):
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(shape=[None,2], dtype=tf.float32)
        Y = tf.placeholder(shape=[None,3], dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32)
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        annealing_step = tf.placeholder(dtype=tf.int32) 
    
        # first hidden layer - conv
        out1 = tf.layers.dense(inputs=X, units=8,activity_regularizer=tf.contrib.layers.l2_regularizer(scale=reguScale))
        if(activation == 'relu'):
            out1=tf.nn.relu(out1)
        elif(activation == 'tanh'):   
            out1=tf.nn.tanh(out1)
        out2 =  tf.layers.dense(inputs=out1, units=16,activity_regularizer=tf.contrib.layers.l2_regularizer(scale=reguScale))
        if(activation == 'relu'):
            out2=tf.nn.relu(out2)
        elif(activation == 'tanh'):   
            out2=tf.nn.tanh(out2)
        out3 =  tf.layers.dense(inputs=out2, units=8,activity_regularizer=tf.contrib.layers.l2_regularizer(scale=reguScale))
        if(activation == 'relu'):
            out3=tf.nn.relu(out3)
        elif(activation == 'tanh'):   
            out3=tf.nn.tanh(out3)            
        out4 = tf.layers.dense(inputs=out3, units=3)
        l2_loss = tf.losses.get_regularization_loss()
        #output layer

        evidence = logits2evidence(out4)
        alpha = evidence + 1
        u = K / tf.reduce_sum(alpha, axis=1, keep_dims=True) #uncertainty
        prob = alpha/tf.reduce_sum(alpha, 1, keepdims=True) 
        loss = tf.reduce_mean(loss_function(Y, alpha, global_step, annealing_step))
        #l2_loss = (tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)) * lmb
        step = tf.train.AdamOptimizer().minimize(loss + l2_loss*lmb, global_step=global_step)
        
        # Calculate accuracy
        pred = tf.argmax(out4, 1)
        truth = tf.argmax(Y, 1)
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
        acc = tf.reduce_mean(match)
        
        total_evidence = tf.reduce_sum(evidence,1, keepdims=True) 
        mean_ev = tf.reduce_mean(total_evidence)
        mean_ev_succ = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*match) / tf.reduce_sum(match+1e-20)
        mean_ev_fail = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
    
        return g, step, X, Y, annealing_step, keep_prob, prob, acc, loss, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail,alpha,total_evidence, out4,truth,pred

def simple_softmax(reguScale=0.005,activation='relu',lmb=None): 
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(shape=[None,2], dtype=tf.float32)
        Y = tf.placeholder(shape=[None,3], dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32)
        annealing_step = tf.placeholder(dtype=tf.int32) 
        # first hidden layer - conv
        out1 = tf.layers.dense(inputs=X, units=8,activity_regularizer=tf.contrib.layers.l2_regularizer(scale=reguScale))
        if(activation == 'relu'):
            out1=tf.nn.relu(out1)
        elif(activation == 'tanh'):   
            out1=tf.nn.tanh(out1)
        out2 =  tf.layers.dense(inputs=out1, units=16,activity_regularizer=tf.contrib.layers.l2_regularizer(scale=reguScale))
        if(activation == 'relu'):
            out2=tf.nn.relu(out2)
        elif(activation == 'tanh'):   
            out2=tf.nn.tanh(out2)
        out3 =  tf.layers.dense(inputs=out2, units=8,activity_regularizer=tf.contrib.layers.l2_regularizer(scale=reguScale))
        if(activation == 'relu'):
            out3=tf.nn.relu(out3)
        elif(activation == 'tanh'):   
            out3=tf.nn.tanh(out3)            
        out4 = tf.layers.dense(inputs=out3, units=3)
        l2_loss = tf.losses.get_regularization_loss()
        prob = tf.nn.softmax(logits=out4) 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out4, labels=Y))
        step = tf.train.AdamOptimizer().minimize(loss + l2_loss)
        # Calculate accuracy
        pred = tf.argmax(out4, 1)
        truth = tf.argmax(Y, 1)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, truth), tf.float32))
        evidence= tf.zeros([3, 4], tf.int32)
        alpha= tf.zeros([3, 4], tf.int32)
        return g, step, X, Y,annealing_step, keep_prob, prob, acc, loss,None, evidence, None, None, None,alpha,None, out4,truth,pred

def predictConf(proba,truth,alpha):
    #this is the correct prediction conf
    res1=np.zeros([proba.shape[0]])
    #this is the wrongly predicted class's conf(mean)
    res2=np.zeros([proba.shape[0]])
    #this is the dirStr(mean) of wrongly predicted class's conf
    res3=np.zeros([proba.shape[0]])
    #this is the dirStr(mean) of wrongly predicted class's alpha (mean)
    res4=np.zeros([proba.shape[0]])
    #convert proba to pred
    pred=np.argmax(proba,axis=1)
    for i in range(proba.shape[0]):
        res1[i]=proba[i,truth[i]]
        if(pred[i]==truth[i]):#then add nothing
            res2[i]=0
            res3[i]=0
            res4[i]=0
        else:
            res2[i]=proba[i,pred[i]]
            res3[i]=np.sum(alpha[i,:])
            res4[i]=alpha[i,pred[i]]
    return [np.mean(res1),np.mean(res2),np.mean(res3),np.mean(res4)]        
   
    
    
def showSubOpinion(alpha,sortInd,top=5,distype='candi'):
    if(distype=='candi'):
        for i in list(range(1,top*2+1)):
            plt.subplot(2,top,i)
            lenArray=np.arange(alpha[sortInd[0],:].shape[0])
            if(i<top+1):
                plt.bar(lenArray,alpha[sortInd[i-1],:])
            else:
                plt.bar(lenArray,alpha[sortInd[-i+4],:])
    else:#plot train(model)'s alpha
        plt.figure()
        plt.bar(np.arange(len(alpha)),alpha)
                
def imgPlot(data, index,shape=[2,5],dim=[28,28]):
    temData=data[index,:].copy()
    plt.figure()
    for i in list(range(1,shape[0]*shape[1]+1)):
        plt.subplot(shape[0],shape[1],i)
        plt.imshow(temData[i-1,:].reshape(dim[0],dim[1]))
        
#count all one alphas        
def countTrivialAlpha(AlphaList):
    tem=np.sum(AlphaList,axis=1)
    print(len(np.where(tem==10)[0]))  
    return(len(np.where(tem==10)[0]))


def getSampleQuality(alpha,type='dirStr'):
    res=np.zeros([alpha.shape[0]])
    for i in range(alpha.shape[0]):
        if (type=='dirStr'):
            res[i]=np.sum(alpha[i,:])
        elif (type=='H_alpha'):#this will normalize alpha automatically thus this is equivalent to use predictive entorpy.
            res[i]=sp.stats.entropy(alpha[i,:])            
        elif (type=='H_mu'):#differential entropy of Dirichlet given alpha
            res[i]=sp.stats.dirichlet.entropy(alpha[i,:])            
    return res

#provide some interesting properties of the AL sample.
def sampeStatAnalysis(alpha,data,trainInd,numClass=10):#used to analysis useful proerties from exausted sampling
    numSamples=len(alpha)
    #see how many opt samples like [1,1,1,1,..1]
    res1=np.zeros([numSamples])
    #see how many opt samples like [100,1,1,..1](single spike)                  
    res2=np.zeros([numSamples])
    for i in range(numSamples):
        res1[i]=np.sum(alpha[i])
        res2[i]=list(alpha[i]).count(1)
        
    print(str(len(np.where(res1==numClass)[0])) + ' samples like [1,1,...1] exist in ' + str(numSamples) + 'samples. Indexed as' + str(np.where(res1==numClass)[0]))
    print(str(len(np.where(res2==numClass-1)[0])) + ' samples like [100,1,...1] (single spike) exist in ' + str(numSamples) + 'samples. Indexed as' + str(np.where(res2==numClass-1)[0]))  
    print('the first row of the next fig is train samples. Index start from the 2nd row')
    imgPlot(data,trainInd,shape=[10,6])
    
#def plotDirEn():
    
#this method add candidateX and its predicted labels to a nn and compute the diff    
def retrainCandidate(X,Y,trainInd,candInd,Y_pre,alpha_pre,diffType='dirStr',epoch=10):
    g2, step2, X2, Y2, annealing_step, keep_prob2, prob2, acc2, loss2, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail,alpha,total_evidence,logits2,tru2,pred2= LeNet_EDL()
    sess2 = tf.Session(graph=g2)
    with g2.as_default():
        sess2.run(tf.global_variables_initializer())
    diff_res=[]
    pre_alpha=[]
    post_alpha=[]
    for candiInd,candidate in enumerate(candInd):
        temTrain=trainInd.copy()
        temTrain.append(candidate)
        Xtrain=X[temTrain,:]
        #print(Y[trainInd,:].shape)
        #print(Y_pre[[candiInd],:].shape)
        Ytrain=np.concatenate((Y[trainInd,:],Y_pre[[candiInd],:]),axis=0)
        kf = KFold(n_splits=10)
        kf.get_n_splits(Xtrain,Ytrain)    
        for epoc in range(epoch):   
            for train_index, batch_index in kf.split(Xtrain):
                #only use test_index
                data=Xtrain[batch_index,:]
                label =Ytrain[batch_index,:]
                feed_dict={X2:data, Y2:label, keep_prob2:.5, annealing_step:5*X.shape[0]}
                sess2.run(step2,feed_dict)
        #get new alpha and compute the difference
        #print(X[[candInd[candiInd]],:].shape)
        #print(Y[[candInd[candiInd]],:].shape)
        alpha_new,evidence_new = sess2.run([alpha,evidence], feed_dict={X2:X[[candInd[candiInd]],:],Y2:Y[[candInd[candiInd]],:],keep_prob2:1.})
        if(diffType=='dirStr'):
            diff=np.sum(np.abs(alpha_pre[candiInd,:]-alpha_new))
            diff_res.append(diff)
            pre_alpha.append(alpha_pre[candiInd,:])
            post_alpha.append(alpha_new)
    return diff_res#,pre_alpha,post_alpha      
 
#add the candidates and their predicted label batchly to the NN for retrain.
def batchRetrainCandidate(X,Y,trainInd,candInd,Y_pre,alpha_pre,diffType='dirStr',epoch=10):
    g2, step2, X2, Y2, annealing_step, keep_prob2, prob2, acc2, loss2, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail,alpha,total_evidence,logits2,tru2,pred2= LeNet_EDL()
    sess2 = tf.Session(graph=g2)
    with g2.as_default():
        sess2.run(tf.global_variables_initializer())
    temTrain=trainInd.copy()
    temTrain=temTrain+candInd
    Xtrain=X[temTrain,:]
    #print(Y[trainInd,:].shape)
    #print(Y_pre[[candiInd],:].shape)
    Ytrain=np.concatenate((Y[trainInd,:],Y_pre[candInd,:]),axis=0)
    kf = KFold(n_splits=10)
    kf.get_n_splits(Xtrain,Ytrain)    
    for epoc in range(epoch):   
        for train_index, batch_index in kf.split(Xtrain):
            #only use test_index
            data=Xtrain[batch_index,:]
            label =Ytrain[batch_index,:]
            feed_dict={X2:data, Y2:label, keep_prob2:.5, annealing_step:5*X.shape[0]}
            sess2.run(step2,feed_dict)
    #get new alpha and compute the difference
    #print(X[[candInd[candiInd]],:].shape)
    #print(Y[[candInd[candiInd]],:].shape)
    alpha_new,evidence_new = sess2.run([alpha,evidence], feed_dict={X2:X[candInd,:],Y2:Y[candInd,:],keep_prob2:1.})
    if(diffType=='dirStr'):
        diff_res=np.sum(np.abs(alpha_pre-alpha_new),axis=1)
    return diff_res#,alpha_pre,alpha_new                 
#imgPlot(X,ind)

#myopically examine the best set of samples. Can be extremely slow. Just for test purpose.
def myopicExaustedTest(trainInd,testInd,candInd,split_batch=10,sample_method='MinDir',AL_iter=20,epoch=15,regLambda=0.005,sampleCriteria='Accu',network=LeNet_EDL,saveFig=1):
    pre_alpha=[]
    post_alpha=[]
    pre_model_alpha=[]
    post_model_alpha=[]
    ALres=[]
    ALresCF=[]
    for ALiter in range(AL_iter):
        print('AL iter:'+str(ALiter))
        g2, step2, X2, Y2, annealing_step, keep_prob2, prob2, acc2, loss2, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail,alpha,total_evidence,logits2,tru2,pred2= network()
        sess2 = tf.Session(graph=g2)
        with g2.as_default():
            sess2.run(tf.global_variables_initializer())
        #start AL testing
            #train model
        Xtest=X[testInd,:]
        Ytest=Y[testInd,:]
        Xtrain=X[trainInd,:]
        Ytrain=Y[trainInd,:]
        for epoc in range(epoch):   
            feed_dict={X2:Xtrain, Y2:Ytrain, keep_prob2:.5, annealing_step:5*X.shape[0]}
            sess2.run(step2,feed_dict)
            
        if(saveFig is not False):
            if(ALiter==0):#true post only need to save once
                plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2],plotType='truePost',plotIter=ALiter);
            elif(ALiter%saveFig==0):                
                plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2],plotType='diffEn',plotIter=ALiter,plotFreq=saveFig,acc=test_acc,cf=test_conf);
                plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2],plotType='predictEn',plotIter=ALiter,plotFreq=saveFig,acc=test_acc,cf=test_conf)
                plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2],plotType='dec',plotIter=ALiter,plotFreq=saveFig,acc=test_acc,cf=test_conf)
                plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2],plotType='invVac',plotIter=ALiter,plotFreq=saveFig,acc=test_acc,cf=test_conf)
                plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2],plotType='Dissonance',plotIter=ALiter,plotFreq=saveFig,acc=test_acc,cf=test_conf)
        kf = KFold(n_splits=split_batch)
        kf.get_n_splits(Xtest,Ytest)       
        tem_test_acc=[]
        tem_test_conf=[]
        for train_index, batch_index in kf.split(Xtest):
            tem_acc,tem_cf,truth= sess2.run([acc2,prob2,tru2], feed_dict={X2:Xtest[batch_index,:],Y2:Ytest[batch_index,:],keep_prob2:1.})
            tem_test_acc.append(tem_acc)
            #tem_test_conf.append(predictConf(tem_cf,truth))
        test_acc=np.mean(tem_test_acc)  
        test_conf=np.mean(tem_test_conf)  
        print('testing: %2.4f' % (test_acc))
        print('testingCF: %2.4f' % (test_conf))
        ALres.append(test_acc)
        ALresCF.append(test_conf)
        #get the model evidence
        train_alpha,train_evidence = sess2.run([alpha,evidence], feed_dict={X2:X[trainInd,:],Y2:Y[trainInd,:],keep_prob2:1.})
        pre_model_alpha.append(np.sum(train_alpha,axis=0))
        #get the pre_alpha of candidates
        candi_alpha,candi_evidence,proba,pre = sess2.run([alpha,evidence,prob2,pred2], feed_dict={X2:X[candInd,:],Y2:Y[candInd,:],keep_prob2:1.})
        #add candidate one by one to see the improvement of the model and record how each candidate affect the accu
        accuImprove=[]
        cfImprove=[]
        #all post alphas of the candidates and trains
        post_alphas=[]
        post_model_alphas=[]
        added_test_accus=[]
        added_test_CF=[]
        for candi in candInd:
            #get the train index with a new candidate data
            temTrainInd=trainInd.copy()
            temTrainInd.append(candi)
            #retain the model
            g2, step2, X2, Y2, annealing_step, keep_prob2, prob2, acc2, loss2, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail,alpha,total_evidence,logits2,tru2,pred2= network()
            sess2 = tf.Session(graph=g2)
            with g2.as_default():
                sess2.run(tf.global_variables_initializer())
            for inepoc in range(epoch):    
                sess2.run(step2, feed_dict={X2:X[temTrainInd,:],Y2:Y[temTrainInd,:],keep_prob2:.5, annealing_step:5*X.shape[0]})
            post_candi_alpha,post_candi_evidence = sess2.run([alpha,evidence], feed_dict={X2:X[[candi],:],Y2:Y[[candi],:],keep_prob2:1.})
            post_alphas.append(post_candi_alpha)
            post_train_alpha,post_train_evidence = sess2.run([alpha,evidence], feed_dict={X2:X[temTrainInd,:],Y2:Y[temTrainInd,:],keep_prob2:1.})
            post_model_alphas.append(np.sum(post_train_alpha,axis=0))
            #get test accu
            tem_test_acc=[]
            tem_test_conf=[]
            for train_index, batch_index in kf.split(Xtest):
                '''
                tem_test_acc.append( sess2.run([acc2], feed_dict={X2:Xtest[batch_index,:],Y2:Ytest[batch_index,:],keep_prob2:1.}))
            test_acc_new=np.mean(tem_test_acc)
            added_test_accus.append(test_acc_new)
            '''
                tem_acc,tem_cf,truth= sess2.run([acc2,prob2,tru2], feed_dict={X2:Xtest[batch_index,:],Y2:Ytest[batch_index,:],keep_prob2:1.})
                tem_test_acc.append(tem_acc)
                #tem_test_conf.append(predictConf(tem_cf,truth))
            test_acc_new=np.mean(tem_test_acc)  
            added_test_accus.append(test_acc_new)
            test_conf_new=np.mean(tem_test_conf) 
            added_test_CF.append(test_conf_new)
            cfImprove.append(test_conf_new-test_conf)                
            accuImprove.append(test_acc_new-test_acc)
        #after test all candidate, myopically choose the one that improve model accu the most
        print('myopic max test accu:'+str(max(added_test_accus)))
        print('myopic max test PU:'+str(max(added_test_CF)))
        if(sampleCriteria=='Accu'):
            criteria=accuImprove
        elif(sampleCriteria=='PU'):
            criteria=cfImprove
        sortInd=np.argsort(criteria)[::-1]             
        pre_alpha.append(candi_alpha[sortInd[0],:])
        post_alpha.append(post_alphas[sortInd[0]])
        post_model_alpha.append(post_model_alphas[sortInd[0]])
        trainInd.append(candInd[sortInd[0]])
        candInd=[x for x in candInd if x!=candInd[sortInd[0]]]
    return pre_alpha,post_alpha,pre_model_alpha,post_model_alpha,ALres,ALresCF
        

def mine1(X,reuse=False,activation=None,reguScale=0.005):
    with tf.variable_scope("mine1", reuse=reuse) as scope:
        out1 = tf.layers.dense(inputs=X, units=8,activity_regularizer=tf.contrib.layers.l2_regularizer(scale=reguScale),activation=activation)
        out2 =  tf.layers.dense(inputs=out1, units=16,activity_regularizer=tf.contrib.layers.l2_regularizer(scale=reguScale),activation=activation)
        out3 =  tf.layers.dense(inputs=out2, units=16,activity_regularizer=tf.contrib.layers.l2_regularizer(scale=reguScale),activation=activation)  
        out4 = tf.layers.dense(inputs=out3, units=3)
        l2_loss = tf.losses.get_regularization_loss()
        return out4,l2_loss


def elu_evidence(logits):
    return tf.nn.elu(logits)
#the input X should be train+candidate.    
def simple_Reg_EDL(logits2evidence=relu_evidence,loss_function=mse_loss, lmb=0.0005,activation=tf.nn.tanh,reguScale=0.005,network=mine1):
    g = tf.Graph()
    with g.as_default():
        #both train and caididate should go through the network
        X = tf.placeholder(shape=[None,2], dtype=tf.float32)
        Y = tf.placeholder(shape=[None,3], dtype=tf.float32)
        #XX is OOD data
        XX =  tf.placeholder(shape=[None,2], dtype=tf.float32)
        
        keep_prob = tf.placeholder(dtype=tf.float32)
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        annealing_step = tf.placeholder(dtype=tf.int32) 

        inDOut4,inD_l2_loss=network(X,activation=activation,reguScale=reguScale)
        ooDOut4,ooDOut_l2_loss=network(XX,activation=activation,reuse=True,reguScale=reguScale)
        
        #output layer
        inD_evidence = logits2evidence(inDOut4)
        inD_alpha = inD_evidence + 1
        ooD_evidence = logits2evidence(ooDOut4)
        ooD_alpha = ooD_evidence + 1
        
        u = K / tf.reduce_sum(ooD_alpha, axis=1, keep_dims=True) #uncertainty
        prob = inD_alpha/tf.reduce_sum(inD_alpha, 1, keepdims=True) 
        #in distribution classification loss
        loss = tf.reduce_mean(loss_function(Y, inD_alpha, global_step, annealing_step))
        #ood loss 
        oodLoss=tf.reduce_sum(u)
        #l2_loss = (tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)) * lmb
        #step = tf.train.AdamOptimizer().minimize(loss + inD_l2_loss +ooDOut_l2_loss - oodLoss*lmb, global_step=global_step)
        step = tf.train.AdamOptimizer().minimize(loss + inD_l2_loss - oodLoss*lmb, global_step=global_step)
        # Calculate accuracy
        pred = tf.argmax(inDOut4, 1)
        truth = tf.argmax(Y, 1)
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
        acc = tf.reduce_mean(match)
        
        total_evidence = tf.reduce_sum(inD_evidence,1, keepdims=True) 
        mean_ev = tf.reduce_mean(total_evidence)
        mean_ev_succ = tf.reduce_sum(tf.reduce_sum(inD_evidence,1, keepdims=True)*match) / tf.reduce_sum(match+1e-20)
        mean_ev_fail = tf.reduce_sum(tf.reduce_sum(inD_evidence,1, keepdims=True)*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
        return g, step, X, Y, XX, annealing_step, keep_prob, prob, acc, loss, u, inD_evidence, mean_ev, mean_ev_succ, mean_ev_fail,inD_alpha,total_evidence, inDOut4,truth,pred

#
#analysis how model performs for OOD
def analysisEn(entropy,proba,OODInd,candInd,trainInd):
    #print('analysis label'+str(np.where(Y[candInd,:]==1)[1][np.argsort(entropy)[::-1][0]]))
    #used to show why softmax perform so well? is it because the all unknow class's instances have high entropy prediction?
    #q1 how many candidate have maximized en?
    entropy=np.array(entropy)
    maxEn=np.max(entropy)
    print('num of max en instance: '+str(len(np.where(entropy==maxEn)[0]))+' with proba:'+str(np.round(proba[np.where(entropy==maxEn)[0]],decimals=3)))
    OODInd=np.array(OODInd)
    OODInd=OODInd-len(trainInd)
    #q2 how many are they from missing class?
    Y1D=np.where(Y[candInd,:]==1)[1]
    maxLabel=list(Y1D[np.where(entropy==maxEn)])
    currentLabel=list(np.unique(np.where(Y[trainInd,:]==1)[1]))
    print('num of max en instance: with unknownclass: '+str(np.sum([1 for x in maxLabel if x not in currentLabel])))
    #q3 what are the top 20 max en label?are they all from unknow class?
    print('current known classes: '+str(list(np.unique(np.where(Y[trainInd,:]==1)[1]))))
    print('the top 10 en instances class:'+str(list(Y1D[np.argsort(entropy)[::-1]][0:10])))
    print('the top 10 en instances en:'+str(list(entropy[np.argsort(entropy)[::-1]][0:10])))
    print('the ood instances class:'+str(list(Y1D[OODInd])))
    print('the ood instances en:'+str(list(entropy[OODInd])))
    #print('the top 5 en instances proba:'+str(list(proba[np.argsort(entropy)[::-1][0:5],:])))
    return np.where(Y[candInd,:]==1)[1][np.argsort(entropy)[::-1][0]]
    
#measures the confidence of wrongly predicted instances. A good uncertainty aware model should have low score.
def rejectTestAccu(proba,true,vacuity,rejectThres=[1.0,0.9,0.8,0.7,0.6,0.5],uncertainType='Vac'):
    pred=np.argmax(proba,axis=1)
    res=[]
    if (uncertainType=='En'):#Then compute the En from proba
        vacuity=[]
        for i in range(proba.shape[0]):    
            vacuity.append(sp.stats.entropy(proba[i,:]))
        #normalize the enropy so it can use same thres list as vacuity
        vacuity=np.array(vacuity)
        vacuity=vacuity/np.sum(vacuity)
    for thres in rejectThres:
        acceptInd=np.where(vacuity<=thres)[0]
        res.append(sk.metrics.accuracy_score(true[acceptInd],pred[acceptInd]))
    return res






#can be used to call EDL and softmax for al experiments.
from sklearn.model_selection import KFold
def AL(X,Y,trainInd,testInd,candInd,split_batch=10,k=3,sample_method='MinDir',AL_iter=50,epoch=15,regLambda=0.005,network=LeNet_EDL,saveFig=False,activation='linear',retrain=False,softmax=False,drop=True,removeTrivial=False,top=0.005,analysisOOD=False,saveDataCode=None):
    global K
    K=k
    model_alpha=[]
    model_alpha_ind=[]
    candidate_alpha=[]
    ALres=[]
    ALres_confident=[]#measure how close is the prediction to the true label.
    ALres_dir=[]
    ALres_alpha=[]
    numOfAllOnes=[]
    AL_rejectACC=[]
    labelAppearIters=[]
    numberOfLabels=len(np.unique(np.where(Y[trainInd,:]==1)[1]))
    testOODVacuity=[]
    testOODInd=getOODInd(np.where(Y[trainInd,:]==1)[1],np.where(Y[testInd,:]==1)[1])#OOD index in test data
    OODDistant=[0]
    OODAplhas=[]
    OODPrediction=[]
    OODTruth=[]
    count=0
    for ALiter in range(AL_iter):
        print('AL iter:'+str(ALiter))
        #if count<(K+20):
        if (ALiter==0 or retrain is True):
            g2, step2, X2, Y2, annealing_step, keep_prob2, prob2, acc2, loss2, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail,alpha,total_evidence,logits2,tru2,pred2 = network(lmb=regLambda,activation=activation)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True        
            sess2 = tf.Session(graph=g2)
            with g2.as_default():
                sess2.run(tf.global_variables_initializer())
        kf = KFold(n_splits=split_batch)
        kf.get_n_splits(X[trainInd,:],Y[trainInd,:])
        #start AL testing
            #train model
        Xtest=X[testInd,:]
        Ytest=Y[testInd,:]
        Xtrain=X[trainInd,:]
        Ytrain=Y[trainInd,:]
        if(drop):
            dropout=0.5
        else:
            dropout=1
        Y_oneDim=np.where(Ytrain==1)[1]
        if(len(np.unique(Y_oneDim))!=numberOfLabels):
            numberOfLabels=len(np.unique(Y_oneDim))
            labelAppearIters.append(ALiter)
            print('###################################instance with new label is added!!!!!############################')
            #since new label is added, update testOODInd
            testOODInd=getOODInd(np.where(Y[trainInd,:]==1)[1],np.where(Y[testInd,:]==1)[1])#OOD index in test data                  
        oodInd,OODDist=OODIdentify(X,trainInd,candInd,type='kernelDensity',top=top,top2=0.5)
        print('OOD Dist(kernel sim): '+str(OODDist))
        print(len(oodInd))
        OODDistant.append(OODDist)
        OODData=X[trainInd+candInd,:][oodInd,:]    
        if(np.log(OODDist)>np.log(OODDistant[-2])):
            count=0
        elif(np.log(OODDist)<=np.log(OODDistant[-2])):
            count+=1    
        print('count: '+str(count))
        for epoc in range(epoch):  
            feed_dict={X2:Xtrain, Y2:Ytrain, keep_prob2:dropout, annealing_step:5*X.shape[0]}
            sess2.run(step2,feed_dict)
            '''
            for train_index, batch_index in kf.split(Xtrain):
                #only use test_index
                data=Xtrain[batch_index,:]
                label =Ytrain[batch_index,:]
                feed_dict={X2:data, Y2:label, keep_prob2:.5, annealing_step:5*X.shape[0]}
                sess2.run(step2,feed_dict)
            '''
        #plot/save the entropy image.
        if(analysisOOD):
        #test OOD's predicted alpha
            OODpred,OODtrue,OODalpha= sess2.run([pred2,tru2,alpha], feed_dict={X2:OODData,Y2:Y[trainInd+candInd,:][oodInd,:],keep_prob2:1.})
            OODAplhas.append(OODalpha)    
            OODPrediction.append(OODpred)
            OODTruth.append(OODtrue)
            #see which OOD predict wrong
            misMatchInd=list(np.where(OODpred!=OODtrue)[0])
            matchInd=list(np.where(OODpred==OODtrue)[0])
            misMatchInd_hAlpha=[x for x in misMatchInd if np.sum(OODalpha[x,:])!=K]
            misMatchInd_lAlpha=[x for x in misMatchInd if np.sum(OODalpha[x,:])==K]
            matchInd_hAlpha=[x for x in matchInd if np.sum(OODalpha[x,:])!=K]
            matchInd_lAlpha=[x for x in matchInd if np.sum(OODalpha[x,:])==K]
        kf = KFold(n_splits=split_batch)
        kf.get_n_splits(Xtest,Ytest)       
        tem_test_acc=[]
        tem_test_wrongconf=[]
        tem_test_wrongDir=[]
        tem_test_wrongAlpha=[]
        tem_rejectTestAccu=[]
        for train_index, batch_index in kf.split(Xtest):
            tem_acc,tem_cf,truth,tem_alpha= sess2.run([acc2,prob2,tru2,alpha], feed_dict={X2:Xtest[batch_index,:],Y2:Ytest[batch_index,:],keep_prob2:1.})
            tem_test_acc.append(tem_acc)
            if(not softmax):
                temRes=predictConf(tem_cf,truth,tem_alpha)
                tem_test_wrongconf.append(temRes[1])
                tem_test_wrongDir.append(temRes[2])
                tem_test_wrongAlpha.append(temRes[3])
                tem_rejectTestAccu.append(np.array(rejectTestAccu(tem_cf,truth,(K/np.sum(tem_alpha,axis=1)),rejectThres=[1.0,0.9,0.8,0.7,0.6,0.5])))
            else:
                temRes=[]
                #anchor4 this should use entropy as threshold but what is the desired threshold?
                tem_rejectTestAccu.append(np.array(rejectTestAccu(tem_cf,truth,None,rejectThres=[1.0,0.9,0.8,0.7,0.6,0.5],uncertainType='En')))
                tem_test_wrongconf.append(0)
                tem_test_wrongDir.append(0)
                tem_test_wrongAlpha.append(0)
        test_rejectAcc=np.mean(np.array(tem_rejectTestAccu),axis=0)
        test_acc=np.mean(tem_test_acc)    
        test_conf=np.mean(tem_test_wrongconf) 
        test_dir=np.mean(tem_test_wrongDir)
        test_alpha=np.mean(tem_test_wrongAlpha)
                #print('epoch %d) '% (epoch+1))
            #train_acc, train_succ, train_fail = sess2.run([acc2,mean_ev_succ,mean_ev_fail], feed_dict={X2:mnist.train.images,Y2:mnist.train.labels,keep_prob2:1.})
        print('testing: %2.4f' % (test_acc))
        print('testing confidence: %2.4f' % (test_conf))
        ALres.append(test_acc)
        ALres_confident.append(test_conf)
        ALres_dir.append(test_dir)
        ALres_alpha.append(test_alpha)
        AL_rejectACC.append(test_rejectAcc)
        if(testOODInd!=0):
            oodX=Xtest[testOODInd,:]
            oodY=Ytest[testOODInd,:]
            ig,oodAlpha= sess2.run([acc2,alpha], feed_dict={X2:oodX,Y2:oodY,keep_prob2:1.})
            testOODVacuity.append(np.sum(K/np.sum(oodAlpha,axis=1)))
        else:
            testOODVacuity.append(0) 
        if(saveFig is not False):
            if(False):#true post only need to save once
                None
                #plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2,logits2],plotType='truePost',plotIter=ALiter);
            elif(ALiter%saveFig==0): 
                cnt=0
                if(not softmax):
                    for pltType in ['predictEn','invVac','Dissonance','showOOD']:
                        plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2,logits2],plotType=pltType,plotIter=ALiter,plotFreq=saveFig,acc=test_acc,cf=test_conf,saveDataCode=saveDataCode+'_'+str(cnt));
                        cnt+=1
                for pltType in ['predictEn','dec']:                       
                    plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2,logits2],plotType=pltType,plotIter=ALiter,plotFreq=saveFig,acc=test_acc,cf=test_conf,saveDataCode=saveDataCode+'_'+str(cnt))
                    cnt+=1
                #predict on candidate
        #print('a1')
        if(not softmax):
            candi_alpha,candi_evidence,proba,pre = sess2.run([alpha,evidence,prob2,pred2], feed_dict={X2:X[candInd,:],Y2:Y[candInd,:],keep_prob2:1.})
            #choose sample with max DirStr
            DirStr=np.sum(candi_alpha,axis=1)
            #total 'counts' of train(model)
            train_alpha,train_evidence = sess2.run([alpha,evidence], feed_dict={X2:X[trainInd,:],Y2:Y[trainInd,:],keep_prob2:1.})
            #train_alpha=train_alpha[0]
            #train_evidence=train_evidence[0]
            total_train_alpha=np.sum(train_alpha,axis=0)
            #recorde model alpha
            model_alpha.append(total_train_alpha)
            model_alpha_ind.append(train_alpha)
            #show how many alpha=[1,1...,1] samples
            print('total number of [1,1,1,...1] samples '+str(len(np.where(np.sum(candi_alpha,axis=1)==K)[0])))
            trivialAlphaInd=np.where(np.sum(candi_alpha,axis=1)==K)[0]
            nontrivialAlphaInd=np.where(np.sum(candi_alpha,axis=1)!=K)[0]
            numOfAllOnes.append(len(trivialAlphaInd))
            if(removeTrivial):
            #then only use non-trivial alphas
                origin_alpha=candi_alpha.copy()
                candi_alpha=candi_alpha[nontrivialAlphaInd,:]
            #sort candiates by DirStr. descendently
        else:#softmax
            proba,pre = sess2.run([prob2,pred2], feed_dict={X2:X[candInd,:],Y2:Y[candInd,:],keep_prob2:1.})
            #candi_alpha,candi_evidence,proba,pre = sess2.run([alpha,evidence,prob2,pred2], feed_dict={X2:Xtrain,Y2:Ytrain,keep_prob2:1.})

        if(sample_method=='MaxDir'):
            sortInd=np.argsort(DirStr)[::-1]
        elif(sample_method=='MinDir'):#equivalent to MAXIMIZE sample Vacuity
            sortInd=np.argsort(DirStr)
        elif(sample_method=='Random'):
            sortInd=[np.random.randint(len(candInd))]
        elif(sample_method=='predic_entropy'):#use predicted label entropy to sample
            entropy=[]
            for probInd in range(proba.shape[0]):
                dataEn=sp.stats.entropy(proba[probInd,:])
                entropy.append(dataEn)                  
            anaRes=analysisEn(entropy,proba,oodInd,candInd,trainInd)    
            sortInd=np.argsort(entropy)[::-1]
            #print('al choosen label:'+str( np.where(Y[candInd,:]==1)[1][sortInd[0]]))
            #if(np.where(Y[candInd,:]==1)[1][sortInd[0]] !=anaRes):
             #   print('AHHHHHHHHHHHHHHHHHHHHH')
              #  return [entropy, Y, candInd, proba, sortInd,anaRes]
        elif(sample_method=='dir_entropy'):
            entropy=np.zeros([candi_alpha.shape[0]])
            for alphaInd in range(candi_alpha.shape[0]):
                entropy[alphaInd]=(sp.stats.dirichlet.entropy(candi_alpha[alphaInd,:]))
            sortInd=np.argsort(entropy)[::-1]
        elif(sample_method=='dissonance'):
            entropy=np.zeros([candi_alpha.shape[0]])
            for alphaInd in range(candi_alpha.shape[0]):
                entropy[alphaInd]=(dissonance(candi_alpha[alphaInd,:]))
            sortInd=np.argsort(entropy)[::-1]                
        elif(sample_method=='DnV'):#F of dissonance and vacuity
            entropy=np.zeros([candi_alpha.shape[0]])
            for alphaInd in range(candi_alpha.shape[0]):
                dissonance_val=(dissonance(candi_alpha[alphaInd,:]))
                vacuity=K/np.sum(candi_alpha[alphaInd,:])
                entropy[alphaInd]=2*(dissonance_val+1e-3)*(vacuity+1e-3)/(dissonance_val+vacuity+1e-6)
            sortInd=np.argsort(entropy)[::-1] 
        elif(sample_method=='DPlusV'):
            entropy=np.zeros([candi_alpha.shape[0]])
            vacc=np.zeros([candi_alpha.shape[0]])
            diss=np.zeros([candi_alpha.shape[0]])
            for alphaInd in range(candi_alpha.shape[0]):
                dissonance_val=dissonance(candi_alpha[alphaInd,:])
                vacuity=K/np.sum(candi_alpha[alphaInd,:])
                vacc[alphaInd]=vacuity
                diss[alphaInd]=dissonance_val
                entropy[alphaInd]=dissonance_val+vacuity
            sortInd=np.argsort(entropy)[::-1] 
            print('max with:'+ str(np.max(entropy))+' diss:'+str(diss[sortInd[0]])+' vacc:'+str(vacc[sortInd[0]]) +'chosen alpha:'+ str(candi_alpha[sortInd[0],:]))
        elif(sample_method=='sp_diff'):#use predicted label to update NN then get the change(later extend to expectation version) of subjuective opinion.
            #get the predicted label mat
            y_pred=np.zeros([len(candInd),Y.shape[1]])
            for row in range(y_pred.shape[0]):
                y_pred[row,pre[row]]=1#note pre is a list.
            diff=retrainCandidate(X,Y,trainInd,candInd,y_pred,candi_alpha,diffType='dirStr')    
            sortInd=np.argsort(diff)[::-1]    
        elif(sample_method=='PreEnTimesDirStr'):#En[y_predict]*DirStr
            entropy=np.zeros([candi_alpha.shape[0]])
            for alphaInd in range(candi_alpha.shape[0]):
                entropy[alphaInd]=(sp.stats.entropy(candi_alpha[alphaInd,:]))       
            EnTimeDirStr= entropy*DirStr
            sortInd=np.argsort(EnTimeDirStr)[::-1]  
        elif(sample_method=='DirStrOverDirEn'): #DirStr/En[mu]    
            entropy=np.zeros([candi_alpha.shape[0]])
            for alphaInd in range(candi_alpha.shape[0]):
                entropy[alphaInd]=(sp.stats.dirichlet.entropy(candi_alpha[alphaInd,:]))
            DirStrOverEn=DirStr/entropy
            sortInd=np.argsort(DirStrOverEn)[::-1]
        elif(sample_method=='DirEnTimesDirStr'): #DirStr*En[mu]    
            entropy=np.zeros([candi_alpha.shape[0]])
            for alphaInd in range(candi_alpha.shape[0]):
                entropy[alphaInd]=(sp.stats.dirichlet.entropy(candi_alpha[alphaInd,:]))
            DirStrOverEn=DirStr*entropy
            sortInd=np.argsort(DirStrOverEn)
        elif(sample_method=='BvSB'):
            None
        if(not softmax):
            #print sample's alpha
            if(removeTrivial):#convert ind back to cand_alpha's ind
                candi_alpha=origin_alpha.copy()
                sortInd=nontrivialAlphaInd[sortInd]           
            print('sample alpha:'+str(list(candi_alpha[sortInd[0],:])))                          
            #save the candidate strength every 10 iterations
            #if(ALiter%10==0):
            candidate_alpha.append([candi_alpha,sortInd])
        #sample using maximum D-sttrength
        trainInd.append(candInd[sortInd[0]])
        candInd=[x for x in candInd if x!=candInd[sortInd[0]]]
    OOD_analysis={'minmaxSim': OODDistant,'OODAlpha':OODAplhas,'OODPredition':OODPrediction,'OODTruth':OODTruth,'TestOODVaccu':testOODVacuity}
    resultList=[ALres,ALres_confident,ALres_dir,ALres_alpha,AL_rejectACC, candidate_alpha, model_alpha,model_alpha_ind,numOfAllOnes,trainInd,candInd,OOD_analysis]
    return resultList
    
    
#removeTrivial: if true, then only sample using those alpha who != all ones.
def ADL(X,Y,trainInd,testInd,candInd,split_batch=10,sample_method='MinDir',AL_iter=50,epoch=15,regLambda=0.005,network=simple_Reg_EDL,network2=mine1,saveFig=False,activation=tf.nn.tanh,retrain=False,reguScale=0.005,pdfs=None,k=3,top=0.005,vacDecay=0,alterWeight=2,removeTrivial=False,analysisOOD=False,saveDataCode=None):
    global K
    K=k
    model_alpha=[]
    model_alpha_ind=[]
    candidate_alpha=[]
    ALres=[]
    ALres_confident=[]#use to measure how close is the prediction to the true label. Now used to measure how certain the wrongly predictions are
    ALres_dir=[]
    ALres_alpha=[]
    numOfAllOnes=[]
    labelAppearIters=[]
    numberOfLabels=len(np.unique(np.where(Y[trainInd,:]==1)[1]))
    testOODVacuity=[]
    testOODInd=getOODInd(np.where(Y[trainInd,:]==1)[1],np.where(Y[testInd,:]==1)[1])#OOD index in test data
    OODDistant=[0]
    OODAplhas=[]
    OODPrediction=[]
    OODTruth=[]
    AL_rejectACC=[]
    delayedDecay=50#iterations need to start the vac decay
    count=0
    for ALiter in range(AL_iter):
        print('AL iter:'+str(ALiter))
        #get the current labels
        Y_oneDim=np.where(Y[trainInd,:]==1)[1]
        #if ((ALiter==0 and len(np.unique(Y_oneDim))==K) or retrain is True):
            #if(len(np.unique(Y_oneDim))<K or ALiter==0):#only retrain the network before classes are fully discovered
        if count<K:
            g2, step2, X2, Y2,XX2, annealing_step, keep_prob2, prob2, acc2, loss2, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail,alpha,total_evidence,logits2,tru2,pred2= network(lmb=regLambda,activation=activation,reguScale=reguScale,network=network2)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess2 = tf.Session(graph=g2)
            with g2.as_default():
                sess2.run(tf.global_variables_initializer())
        kf = KFold(n_splits=split_batch)
        kf.get_n_splits(X[trainInd,:],Y[trainInd,:])
        #start AL testing
            #train model
        Xtest=X[testInd,:]
        Ytest=Y[testInd,:]
        Xtrain=X[trainInd,:]
        Ytrain=Y[trainInd,:]
        if(len(np.unique(Y_oneDim))!=numberOfLabels):
            numberOfLabels=len(np.unique(Y_oneDim))
            labelAppearIters.append(ALiter)
            print('###################################instance with new label is added!!!!!############################')
            #since new label is added, update testOODInd
            testOODInd=getOODInd(np.where(Y[trainInd,:]==1)[1],np.where(Y[testInd,:]==1)[1])#OOD index in test data                  
        oodInd,OODDist=OODIdentify(X,trainInd,candInd,type='kernelDensity2',top=top,top2=0.5)
        
        print('OOD Dist(kernel sim): '+str(OODDist)+' OOD Num:'+ str(len(oodInd)))
        OODDistant.append(OODDist)
        OODData=X[trainInd+candInd,:][oodInd,:]
        for epoc in range(epoch):   
            feed_dict={X2:Xtrain, Y2:Ytrain,XX2:OODData, keep_prob2:.5, annealing_step:50*X.shape[0]}
            sess2.run(step2,feed_dict)
        #plot/save the entropy image.
        kf = KFold(n_splits=split_batch)
        kf.get_n_splits(Xtest,Ytest)       
        tem_test_acc=[]
        tem_test_wrongconf=[]
        tem_test_wrongDir=[]
        tem_test_wrongAlpha=[]
        tem_rejectTestAccu=[]
        for train_index, batch_index in kf.split(Xtest):
            tem_acc,tem_cf,truth,tem_alpha= sess2.run([acc2,prob2,tru2,alpha], feed_dict={X2:Xtest[batch_index,:],Y2:Ytest[batch_index,:],keep_prob2:1.})
            tem_test_acc.append(tem_acc)
            temRes=predictConf(tem_cf,truth,tem_alpha)
            tem_rejectTestAccu.append(np.array(rejectTestAccu(tem_cf,truth,(K/np.sum(tem_alpha,axis=1)),rejectThres=[1.0,0.9,0.8,0.7,0.6,0.5])))
            tem_test_wrongconf.append(temRes[1])
            tem_test_wrongDir.append(temRes[2])
            tem_test_wrongAlpha.append(temRes[3])
        test_acc=np.mean(tem_test_acc)    
        test_rejectAcc=np.mean(np.array(tem_rejectTestAccu),axis=0)
        test_conf=np.mean(tem_test_wrongconf) 
        test_dir=np.mean(tem_test_wrongDir)
        test_alpha=np.mean(tem_test_wrongAlpha)
        print('testing: %2.4f' % (test_acc))
        print('testing confidence: %2.4f' % (test_conf))
        ALres.append(test_acc)
        ALres_confident.append(test_conf)
        ALres_dir.append(test_dir)
        ALres_alpha.append(test_alpha)
        AL_rejectACC.append(test_rejectAcc)
        if(analysisOOD):
            #test OOD's predicted alpha
            OODpred,OODtrue,OODalpha= sess2.run([pred2,tru2,alpha], feed_dict={X2:OODData,Y2:Y[trainInd+candInd,:][oodInd,:],keep_prob2:1.})
            OODAplhas.append(OODalpha)    
            OODPrediction.append(OODpred)
            OODTruth.append(OODtrue)
            #see which OOD predict wrong
            misMatchInd=list(np.where(OODpred!=OODtrue)[0])
            matchInd=list(np.where(OODpred==OODtrue)[0])
            misMatchInd_hAlpha=[x for x in misMatchInd if np.sum(OODalpha[x,:])!=K]
            misMatchInd_lAlpha=[x for x in misMatchInd if np.sum(OODalpha[x,:])==K]
            matchInd_hAlpha=[x for x in matchInd if np.sum(OODalpha[x,:])!=K]
            matchInd_lAlpha=[x for x in matchInd if np.sum(OODalpha[x,:])==K]
        if(testOODInd!=0):
            oodX=Xtest[testOODInd,:]
            oodY=Ytest[testOODInd,:]
            ig,oodAlpha= sess2.run([acc2,alpha], feed_dict={X2:oodX,Y2:oodY,keep_prob2:1.})
            testOODVacuity.append(np.sum(K/np.sum(oodAlpha,axis=1)))
        else:
            testOODVacuity.append(0)            
        if(saveFig is not False):
            if(False):#true post only need to save once
                None
                #plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2,logits2],plotType='truePost',plotIter=ALiter);
            elif(ALiter%saveFig==0): 
                #use current iter as the name
                cnt=0
                for pltType in ['predictEn','dec','invVac','Dissonance']:#['predictEn','dec','diffEn','invVac','Dissonance','showOOD']:
                    plot2D(X,Y,truePdfs=pdfs,trainInd=trainInd,testInd=testInd,candInd=candInd,model=[sess2,alpha,prob2,X2,Y2,keep_prob2,logits2],plotType=pltType,plotIter=ALiter,plotFreq=saveFig,acc=test_acc,cf=test_conf,saveDataCode=saveDataCode+'_'+str(ALiter)+'_'+str(cnt));
                    cnt+=1
        candi_alpha,candi_evidence,proba,pre = sess2.run([alpha,evidence,prob2,pred2], feed_dict={X2:X[candInd,:],Y2:Y[candInd,:],keep_prob2:1.})
        DirStr=np.sum(candi_alpha,axis=1)
        #total 'counts' of train(model)
        train_alpha,train_evidence = sess2.run([alpha,evidence], feed_dict={X2:X[trainInd,:],Y2:Y[trainInd,:],keep_prob2:1.})
        #train_alpha=train_alpha[0]
        #train_evidence=train_evidence[0]
        total_train_alpha=np.sum(train_alpha,axis=0)
        #recorde model alpha
        model_alpha.append(total_train_alpha)
        model_alpha_ind.append(train_alpha)
        #show how many alpha=[1,1...,1] samples
        print('total number of [1,1,1,...1] samples '+str(len(np.where(np.sum(candi_alpha,axis=1)==K)[0])))
        trivialAlphaInd=np.where(np.sum(candi_alpha,axis=1)==K)[0]
        nontrivialAlphaInd=np.where(np.sum(candi_alpha,axis=1)!=K)[0]
        numOfAllOnes.append(len(trivialAlphaInd))
        if(removeTrivial):
            #then only use non-trivial alphas
            origin_alpha=candi_alpha.copy()
            candi_alpha=candi_alpha[nontrivialAlphaInd,:]
        #sort candiates by DirStr. descendently
        if(sample_method=='MaxDir'):
            sortInd=np.argsort(DirStr)[::-1]
        elif(sample_method=='predic_entropy'):#use predicted label entropy to sample
            entropy=[]
            for probInd in range(proba.shape[0]):
                entropy.append(sp.stats.entropy(proba[probInd,:]))
            sortInd=np.argsort(entropy)[::-1]
        elif(sample_method=='dissonance'):
            entropy=np.zeros([candi_alpha.shape[0]])
            for alphaInd in range(candi_alpha.shape[0]):
                entropy[alphaInd]=(dissonance(candi_alpha[alphaInd,:]))
            sortInd=np.argsort(entropy)[::-1]                
        elif(sample_method=='DPlusV'):#sum of dissonance and decay vacuity
            entropy=np.zeros([candi_alpha.shape[0]])
            vacc=np.zeros([candi_alpha.shape[0]])
            diss=np.zeros([candi_alpha.shape[0]])
            if(np.log(OODDist)>np.log(OODDistant[-2])):
                count=0
            elif(np.log(OODDist)<=np.log(OODDistant[-2])):
                count+=1      
            print('count:'+str(count))
            for alphaInd in range(candi_alpha.shape[0]):
                dissonance_val=dissonance(candi_alpha[alphaInd,:])
                vacuity=K/np.sum(candi_alpha[alphaInd,:])
                vacc[alphaInd]=vacuity
                diss[alphaInd]=dissonance_val
                if(vacDecay=='minmaxGuided'):
                    #if it is <0.3, fully weighted(more likely to choose all one alphas)
                    if(OODDist<0.3):
                        entropy[alphaInd]=dissonance_val+vacuity
                    else:
                        entropy[alphaInd]=dissonance_val+(1-OODDist)*vacuity
                elif(vacDecay=='discoverClassGuided'):
                    if(numberOfLabels<K or ALiter<delayedDecay):#This is kind of cheat for we do not know K if we assume some class samples are missing from initial training set
                        entropy[alphaInd]=dissonance_val+vacuity
                    else:#after all class been discovered reduce vacuity (use fix ratio )    
                        entropy[alphaInd]=dissonance_val+(1- 0.002*ALiter/K)*vacuity
                elif(vacDecay=='logMinVal'):
                    if(count>K/2):
                        entropy[alphaInd]=dissonance_val+(1- 0.002*ALiter/K)*vacuity
                    else:
                        entropy[alphaInd]=dissonance_val+vacuity              
                else:#else if vacDecay is a real value
                    entropy[alphaInd]=dissonance_val+(1- vacDecay*ALiter/K)*vacuity
            sortInd=np.argsort(entropy)[::-1] 
            print('max with:'+ str(np.max(entropy))+' diss:'+str(diss[sortInd[0]])+' vacc:'+str(vacc[sortInd[0]]) +'chosen alpha:'+ str(candi_alpha[sortInd[0],:]))
        elif(sample_method=='EPlusV'):#sum of dissonance and decay vacuity
            #entropy=np.zeros([candi_alpha.shape[0]])
            ourRes.append(AL_OOD_EDL(X,Y,trainInd,testInd,candInd,split_batch=10,sample_method='DPlusV',AL_iter=300,epoch=800,regLambda=0.001,network=LeNet_Reg_EDL,saveFig=False,activation=tf.nn.tanh,retrain=True,reguScale=0.03,pdfs=None,k=10,network2=mine2,top=0.004,analysisOOD=True,vacDecay='discoverClassGuided'))
            vacc=np.zeros([candi_alpha.shape[0]])
            diss=np.zeros([candi_alpha.shape[0]])
            en=np.zeros([candi_alpha.shape[0]])
            for alphaInd in range(candi_alpha.shape[0]):
                dissonance_val=dissonance(candi_alpha[alphaInd,:])
                vacuity=K/np.sum(candi_alpha[alphaInd,:])
                en_val=sp.stats.entropy(candi_alpha[alphaInd,:])
                vacc[alphaInd]=vacuity
                diss[alphaInd]=dissonance_val
                en[alphaInd]=en_val
                if(vacDecay=='minmaxGuided'):
                    #if it is <0.3, fully weighted(more likely to choose all one alphas)
                    if(OODDist<0.3):
                        entropy[alphaInd]=en_val+vacuity
                    else:
                        entropy[alphaInd]=en_val+(1-OODDist)*vacuity
                elif(vacDecay=='discoverClassGuided'):
                    if(numberOfLabels<K):#This is kind of cheat for we do not know K if we assume some class samples are missing from initial training set
                        entropy[alphaInd]=en_val+vacuity
                    else:#after all class been discovered reduce vacuity (use fix ratio )    
                        entropy[alphaInd]=en_val+(1- 0.005*ALiter/K)*vacuity
                else:#else if vacDecay is a real value
                    entropy[alphaInd]=en_val+(1- vacDecay*ALiter/K)*vacuity
            sortInd=np.argsort(entropy)[::-1] 
            print(' diss:'+str(diss[sortInd[0]])+' vacc:'+str(vacc[sortInd[0]]) + 'en: ' + str(en[sortInd[0]])+'chosen alpha:'+ str(candi_alpha[sortInd[0],:]))            
        elif(sample_method=='sp_diff'):#use predicted label to update NN then get the change(later extend to expectation version) of subjuective opinion.
            #get the predicted label mat
            y_pred=np.zeros([len(candInd),Y.shape[1]])
            for row in range(y_pred.shape[0]):
                y_pred[row,pre[row]]=1#note pre is a list.
            diff=retrainCandidate(X,Y,trainInd,candInd,y_pred,candi_alpha,diffType='dirStr')    
            sortInd=np.argsort(diff)[::-1]    
        elif(sample_method=='DPlusVPlusE'):
            entropy=np.zeros([candi_alpha.shape[0]])
            vacc=np.zeros([candi_alpha.shape[0]])
            diss=np.zeros([candi_alpha.shape[0]])
            for alphaInd in range(candi_alpha.shape[0]):
                dissonance_val=dissonance(candi_alpha[alphaInd,:])
                vacuity=K/np.sum(candi_alpha[alphaInd,:])
                en=sp.stats.entropy(candi_alpha[alphaInd,:])
                vacc[alphaInd]=vacuity
                diss[alphaInd]=dissonance_val
                entropy[alphaInd]=dissonance_val+vacuity+en
            sortInd=np.argsort(entropy)[::-1]
        elif(sample_method=='alter'):
                #sample using vac and diss round robin
                entropy=np.zeros([candi_alpha.shape[0]])
                vacc=np.zeros([candi_alpha.shape[0]])
                diss=np.zeros([candi_alpha.shape[0]])
                for alphaInd in range(candi_alpha.shape[0]):
                    dissonance_val=dissonance(candi_alpha[alphaInd,:])
                    vacuity=K/np.sum(candi_alpha[alphaInd,:])
                    vacc[alphaInd]=vacuity
                    diss[alphaInd]=dissonance_val
                    if(ALiter%alterWeight==0):
                        entropy[alphaInd]=vacuity
                    else:
                        entropy[alphaInd]=dissonance_val
                sortInd=np.argsort(entropy)[::-1] 
                print('max with:'+ str(np.max(entropy))+' diss:'+str(diss[sortInd[0]])+' vacc:'+str(vacc[sortInd[0]]) +'chosen alpha:'+ str(candi_alpha[sortInd[0],:]))
        if(removeTrivial):#convert ind back to cand_alpha's ind
            candi_alpha=origin_alpha.copy()
            sortInd=nontrivialAlphaInd[sortInd]                    
        #print sample's alpha
        print('sample alpha:'+str(list(candi_alpha[sortInd[0],:])))                          
        #save the candidate strength every 10 iterations
        #if(ALiter%10==0):
        candidate_alpha.append([candi_alpha,sortInd])
        #sample using maximum D-sttrength
        trainInd.append(candInd[sortInd[0]])
        candInd=[x for x in candInd if x!=candInd[sortInd[0]]]
    print('new labels are added at iters:'+str(labelAppearIters))
    OOD_analysis={'minmaxSim': OODDistant,'OODAlpha':OODAplhas,'OODPredition':OODPrediction,'OODTruth':OODTruth,'TestOODVaccu':testOODVacuity}
    resultList=[ALres,ALres_confident,ALres_dir,ALres_alpha,AL_rejectACC, candidate_alpha, model_alpha,model_alpha_ind,numOfAllOnes,trainInd,candInd,labelAppearIters,OOD_analysis]
    return resultList