#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split


# In[123]:


def train_model(input_var,algo):
    
    
    heart=pd.read_csv('C:/Users/Shreyansh/.spyder-py3/heart.csv')
    X=heart[['ca','cp','exang','oldpeak','sex','thal','thalach']]
    y=heart['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    if(algo=='SVM'):
        target_predicted=SVM(X_train,y_train,X_test,y_test,input_var)
        
    elif(algo=='KNN'):
        target_predicted=KNN(X_train,y_train,X_test,y_test,input_var)
    else:
        target_predicted=LR(X_train,y_train,X_test,y_test,input_var)
    return target_predicted

def f_score_calculation(confusion_matrix):
    tp=confusion_matrix[0][0]
    fp=confusion_matrix[0][1]
    fn=confusion_matrix[1][0]
    tn=confusion_matrix[1][1]
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f_score = 2*recall*precision/(recall+precision)
    return f_score

def auc_curve(predictions,y_test):  
    
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    return roc_auc
    
    
def SVM(X_train,y_train,X_test,y_test,input_var):
    svc = svm.SVC(kernel='rbf',C=10) # Check by varying C=10,100,1000
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    auc_plot=auc_curve(y_pred,y_test)
    target_predicted = svc.predict(input_var)
    conf_mat = confusion_matrix(y_test, y_pred)
    f_score=f_score_calculation(conf_mat)
    
    return target_predicted,f_score,conf_mat,auc_plot,y_pred,y_test
    
def KNN(X_train,y_train,X_test,y_test,input_var):   
    knn = KNeighborsClassifier(n_neighbors=3)  
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    auc_plot=auc_curve(y_pred,y_test)
    target_predicted = knn.predict(input_var)
    conf_mat = confusion_matrix(y_test, y_pred)
    f_score=f_score_calculation(conf_mat)
    return target_predicted,f_score,conf_mat,auc_plot,y_pred,y_test
   
def LR(X_train,y_train,X_test,y_test,input_var):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred=logreg.predict(X_test)
    auc_plot=auc_curve(y_pred,y_test)
    target_predicted = logreg.predict(input_var)
    conf_mat = confusion_matrix(y_test, y_pred)
    f_score=f_score_calculation(conf_mat)
    return target_predicted,f_score,conf_mat,auc_plot,y_pred,y_test

