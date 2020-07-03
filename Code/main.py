#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:10:21 2019

@author: Akshatha Shivashankar Chindalur
"""
# importing the required libraries for sentiment analysis
import pandas as pd 

from preprocess import preprocess
from model import feature_select, build_model

# loading the test and train datasets
train_data = pd.read_csv('./train_data.csv')
train_label = pd.read_csv('./train_label.csv')
test_data = pd.read_csv('./test_data.csv')

# preprocessing steps applied to training data
train_X = preprocess(train_data['text'])  
print("train preprocess done!")

# preprocessing steps applied to test data
test_X = preprocess(test_data['text'])
print("test preprocess done!")

# feature selection
train_features, test_features = feature_select(train_X, test_X)

# training a machine learning model
predicted_labels = build_model(train_features, train_label['label'], test_features)

# saving the results obtained from the analyser
test_labels = {'test_id': test_data['test_id'], 'label': predicted_labels}
test_df = pd.DataFrame(test_labels, columns=['test_id','label'])
test_df.to_csv('predict_label.csv', index=False)