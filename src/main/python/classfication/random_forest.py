# coding=utf-8
'''
Created on 2016年5月3日

@author: star
'''

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
import csv
import time
import numpy as np
import sys
import os

import dataset.get_dataset

def train(X, y):
    print 'Train Random Forest model...'
    # X = preprocessing.scale(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
    # model = SVC(C = 1)
    # model = LinearSVC(C=1)
    model = RandomForestClassifier(n_estimators = 100)
    # model = LogisticRegression(C = 1,penalty = 'l2', tol = 0.001, max_iter = 20000)
    # model.fit(X_train, y_train)
    model.fit(X, y)
    # y_pred = model.predict(X_test)
    # count = 0
    # pos_count = 0
    # for i in range(len(y_pred)):
        # if y_test[i]==1 and y_pred[i] == 1: count += 1
        # if y_test[i] == 1: pos_count += 1
        # if y_pred[i] == y_test[i]: count += 1
    # print count
    # print float(count)/len(y_pred)
    # print pos_count
    # print float(count)/pos_count
    return model



if __name__ == '__main__':
    train_dataset_path = '/home/hadoop/aliMatch/data/mars_tianchi_train_dataset.csv'
    test_dataset_path = '/home/hadoop/aliMatch/data/mars_tianchi_test_dataset.csv'
    train_file_path = '/home/hadoop/aliMatch/data/mars_tianchi_train_data.csv'
    test_file_path = '/home/hadoop/aliMatch/data/mars_tianchi_test_data.csv'
    predict_file_path = '/home/hadoop/aliMatch/data/mars_tianchi_predict_data.csv'
    predict_days  = 60
    artist_num = 50

    (train_data_predicts, test_data_preticts) = dataset.get_dataset.get_data_preticts(train_file_path, test_file_path)
    train_dataset = open(train_dataset_path, 'r')
    test_dataset = open(test_dataset_path, 'r')
    X = []
    y = []
    for line in train_dataset:
        line = line.strip().split(',')
        y.append(int(line[0]))
        X.append(map(float, line[1:]))
    model = train(X, y)
    X_test = []
    y_test = []
    res = []
    positive_num = []
    for i in range(artist_num):
	    res.append([])
	    positive_num.append(0)
	    for j in range(predict_days):
	        res[i].append(0)
    print 'Test Random Forest model...'
    for line in test_dataset:
        
        line = line.strip().split(',')
        y_num = int(line[0])
        x_num = int(line[1])
        X_test = map(float, line[3:])
        y_pred = model.predict([X_test])
        if y_pred[0] < 1:
            continue
        positive_num[y_num] = positive_num[y_num] + 1
        for i in range(predict_days):
		    res[y_num][i] += min(train_data_predicts[x_num][i+2],3.0)
    print 'Output result...'
    predict_file= csv.writer(file(predict_file_path, 'wb'))
    for artist_id in range(artist_num):
	    mid = test_data_preticts[artist_id][0]
	    avr = test_data_preticts[artist_id][1]
	    std_predict = []
	    for i in range(predict_days):
	        std_predict.append(test_data_preticts[artist_id][i+2]*avr)
	    my_predict = []
	    if positive_num[artist_id] < 1:
	        for i in range(predict_days):
	            my_predict.append(mid)
	    else:
	        for i in range(predict_days):
	            my_predict.append(avr*res[artist_id][i]/positive_num[artist_id])
	    predict_file.writerow([str(col) for col in std_predict])
	    predict_file.writerow([str(col) for col in my_predict])
