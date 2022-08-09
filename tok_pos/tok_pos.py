#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:09:07 2022

@author: yuyingren
"""
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import statistics
import os
import re
import jieba.posseg as pseg


def tok_pos(sentence: str):
    

    tokens = list(sentence)
    token_pos=[]
    for i in tokens:
            t_pos = pseg.cut(i)
            for j in t_pos:
                token_pos.append(tuple(j))
    return token_pos



def extract_features(char, sen):
    
    token_sen = tok_pos(sen)


    keys = ['tok_tag', 'tok-1_tag','tok+1_tag']
    features = {}.fromkeys(keys, '')
    
    for (i, tok_tag) in enumerate(token_sen):
        if char in tok_tag[0]:
            tar_idx = i

            features['tok_tag'] = token_sen[i][1]
            if tar_idx - 1 < 0:
                features['tok-1'] = ''
                features['tok-1_tag'] = ''
            else:
                features['tok-1'] = token_sen[tar_idx - 1][0]
                features['tok-1_tag'] = token_sen[tar_idx - 1][1]

            if tar_idx - 2 < 0:
                features['tok-2'] = ''
                features['tok-2_tag'] = ''
            else:
                features['tok-2'] = token_sen[tar_idx - 2][0]
                features['tok-2_tag'] = token_sen[tar_idx - 2][1]

            if tar_idx + 1 >= len(token_sen):
                features['tok+1'] = ''
                features['tok+1_tag'] = ''
            else:
                features['tok+1'] = token_sen[tar_idx + 1][0]
                features['tok+1_tag'] = token_sen[tar_idx + 1][1]
            
            if tar_idx + 2 >= len(token_sen):
                features['tok+2'] = ''
                features['tok+2_tag'] = ''
            else:
                features['tok+2'] = token_sen[tar_idx + 2][0]
                features['tok+2_tag'] = token_sen[tar_idx + 2][1]
                    
    return features


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

def read_file(file_path: str):
    labels = []
    features=[]
    with open(file_path, 'r') as file:
        
        for i in file:
            line = i.rstrip().split('\t')
            char = line[0]
            label = re.sub("'", "", line[1])
            labels.append(label)
            sentence = line[2]
            feature = extract_features(char, sentence)
            features.append(feature)
            
    return labels, features


def main(path_train: str, path_test: str):

    dir_train = os.listdir(path_train)
    dir_test = os.listdir(path_test)

    train_test = []

    for i in dir_train:
        if i.startswith('.'):
            continue
        for j in dir_test:
            if j.startswith('.'):
                continue
            if i == j:
                i = os.path.join(path_train, i)
                j = os.path.join(path_test, j)
                train_test.append((i, j))
                
    res_rfc = []
    res_svm = []
    res_lr = []         
    for i in train_test:

        TrainLabels = read_file(i[0])[0]
        TrainFeatures = read_file(i[0])[1]
        TestLabels = read_file(i[1])[0]
        TestFeatures = read_file(i[1])[1]
        
        resRFC = train_predict_rfc(TrainFeatures, TrainLabels, TestFeatures, TestLabels)
        res_rfc.append(resRFC)
        resSVM = train_predict_svm(TrainFeatures, TrainLabels, TestFeatures, TestLabels)
        res_svm.append(resSVM)
        resLR = train_predict_lr(TrainFeatures, TrainLabels, TestFeatures, TestLabels)
        res_lr.append(resLR)
        
    with open("results_TokPos.tsv", "w") as resfile:
        
        print("RandomForest: " + "\t" + str(statistics.mean(res_rfc)), file = resfile)
            
        print("SVM: " + "\t" + str(statistics.mean(res_svm)), file = resfile)
            
        print("Logistic Regression: " + "\t" + str(statistics.mean(res_lr)), file = resfile)
        
    print("ok")
                
main("data/train", "data/test")


    

            
