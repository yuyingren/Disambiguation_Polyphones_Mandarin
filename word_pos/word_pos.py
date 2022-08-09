#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 00:30:56 2022

@author: yuyingren
"""

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import statistics


        
        
def train_predict_rfc(train_features: dict, train_labels: list,
                      test_features: dict, test_labels: list):
    
    vec = DictVectorizer()
    train_vec = vec.fit_transform(train_features) 


    rfc = RandomForestClassifier()
    
    rfc.fit(train_vec, train_labels)
    test_vec = vec.transform(test_features)
    prediction = rfc.predict(test_vec)
    

    pred_gold = zip(prediction, test_labels)
    
    correct = 0

    for pred, gold in pred_gold:
        if pred == gold:
            correct += 1
    accuracy = correct/len(test_labels)
    
    return accuracy

def train_predict_svm(train_features: dict, train_labels: list,
                      test_features: dict, test_labels: list):

    vec = DictVectorizer()
    train_vec = vec.fit_transform(train_features) 

    svm = SVC()
    svm.fit(train_vec, train_labels)
    test_vec = vec.transform(test_features)
    prediction = svm.predict(test_vec)
    
    
    pred_gold = zip(prediction, test_labels)
    
    correct = 0
  
    for pred, gold in pred_gold:
        if pred == gold:
            correct += 1
    accuracy = correct/len(test_labels)
    return accuracy

def train_predict_lr(train_features: dict, train_labels: list,
                      test_features: dict, test_labels: list):
    
    
    vec = DictVectorizer()
    train_vec = vec.fit_transform(train_features) 


    lr_model = LogisticRegression(penalty = "l1", C = 10, solver = "liblinear")
    lr_model.fit(train_vec, train_labels)
    test_vec = vec.transform(test_features)
    prediction = lr_model.predict(test_vec)
    
    pred_gold = zip(prediction, test_labels)
    
    correct = 0
  
    for pred, gold in pred_gold:
        if pred == gold:
            correct += 1
    accuracy = correct/len(test_labels)
    
    return accuracy

def main():

    TrainLabels = []
    TrainFeatures = []
    TestLabels = []
    TestFeatures = []
    with open("labels_features/train_labels.txt", "r") as tl:
        for  i in tl:
            TrainLabels.append(eval(i.rstrip()))
            
    with open("labels_features/train_features.txt", "r") as tf:
        for  i in tf:
            TrainFeatures.append(eval(i.rstrip()))
            
    with open("labels_features/test_labels.txt", "r") as tel:
        for  i in tel:
            TestLabels.append(eval(i.rstrip()))
        
    with open("labels_features/test_features.txt", "r") as tef:
        for  i in tef:
            TestFeatures.append(eval(i.rstrip()))
            
    rfc_res = []
    lr_res = []
    svm_res = []
    for i in zip(TrainFeatures, TrainLabels, TestFeatures, TestLabels):
        rfc_res.append(train_predict_rfc(i[0], i[1], i[2], i[3]))
        svm_res.append(train_predict_svm(i[0], i[1], i[2], i[3]))
        lr_res.append(train_predict_lr(i[0], i[1], i[2], i[3]))
    
    with open("results_WordPos.tsv", "w") as resfile:
        
        print("RandomForest: " + "\t" + str(statistics.mean(rfc_res)), file = resfile)
        
        print("SVM: " + "\t" + str(statistics.mean(svm_res)), file = resfile)
        
        print("Logistic Regression: " + "\t" + str(statistics.mean(lr_res)), file = resfile)
        
    
main()