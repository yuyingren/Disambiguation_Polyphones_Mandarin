#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuyingren
"""
import os
import re
import jieba.posseg as pseg


def word_pos(sentence: str):
    
    word_pos=[]
    sent = pseg.cut(sentence)
    for i in sent:
        word_pos.append(tuple(i))
        
    return word_pos


def extract_features(char, sen):
    
    word_sen = word_pos(sen)
    # print(word_sen)

    keys = ['tok_tag', 'tok-1_tag','tok+1_tag']
    features = {}.fromkeys(keys, '')
    
    for (i, tok_tag) in enumerate(word_sen):
        if char in tok_tag[0]:
            tar_idx = i

            features['tok_tag'] = word_sen[i][1]
            if tar_idx - 1 < 0:
                features['tok-1'] = ''
                features['tok-1_tag'] = ''
            else:
                features['tok-1'] = word_sen[tar_idx - 1][0]
                features['tok-1_tag'] = word_sen[tar_idx - 1][1]

            if tar_idx - 2 < 0:
                features['tok-2'] = ''
                features['tok-2_tag'] = ''
            else:
                features['tok-2'] = word_sen[tar_idx - 2][0]
                features['tok-2_tag'] = word_sen[tar_idx - 2][1]

            if tar_idx + 1 >= len(word_sen):
                features['tok+1'] = ''
                features['tok+1_tag'] = ''
            else:
                features['tok+1'] = word_sen[tar_idx + 1][0]
                features['tok+1_tag'] = word_sen[tar_idx + 1][1]
            
            if tar_idx + 2 >= len(word_sen):
                features['tok+2'] = ''
                features['tok+2_tag'] = ''
            else:
                features['tok+2'] = word_sen[tar_idx + 2][0]
                features['tok+2_tag'] = word_sen[tar_idx + 2][1]
                    
    return features


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
                
    with open("labels_features/train_labels.txt", "w") as tl:     
        for i in train_test:
            print(read_file(i[0])[0], file = tl)
            
    print("ok")
    
    with open("labels_features/train_features.txt", "w") as tf:     
        for i in train_test:
            print(read_file(i[0])[1], file = tf)
            
    print("ok")
    
    with open("labels_features/test_labels.txt", "w") as tel:     
        for i in train_test:
            print(read_file(i[1])[0], file = tel)
            
    print("ok")
    
    with open("labels_features/test_features.txt", "w") as tef:     
        for i in train_test:
            print(read_file(i[1])[1], file = tef)
            
    print("ok")        
            
main("data/train", "data/test")