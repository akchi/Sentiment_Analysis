#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:52:43 2019

@author: Akshatha Shivashankar Chindalur
"""
# importing the required libraries for training the model
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression

#   This function builds a machine learning model. A logistic regression
#   classifier is implemented from the features and labels of the training
#   dataset.
#
#   :param X:    the list of features obtained from the training data
#          y:    the list of class labels of the corresponding reviews in X
#          test: list of features extracted from the test data
#   :return y_test_labels: a list of predicted class labels of the reviews 
#                          in the test data.

def build_model(X, y, test):    

    log_model = LogisticRegression(random_state=0)
    log_model.fit(X,y)
    
    y_test_labels = log_model.predict(test)
    
    return y_test_labels

#   This function extracts the required features for the sentiment analyser.
#   The function extracts a set of suitable features from the training data 
#   and a similar set of features are extracted from the test data. (The 
#   initial extraction is independent of the test data).
#
#   :param train: a list containing the preprocessed training reviews.
#          test: a list containing the preprocessed testing reviews.
#   :return train_features: list of features extracted for the training data.
#           test_features: list of features extracted for the testing data.
def feature_select(train, test):
    
    tf_idf = TfidfVectorizer(ngram_range=(1,2))
    
    train_features = tf_idf.fit_transform(train)
    
    test_features = tf_idf.transform(test)
    
    return train_features, test_features