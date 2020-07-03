#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:17:08 2019

@author: Akshatha Shivashankar Chindalur
"""
#   importing the natural language processing toolkit.
import nltk  

#   This function performs the intial pre-processing task required 
#   for model building.
#
#   :param corpus: the list containing the reviews to be analysed.
#   :return processed: the list of all pre-processed reviews that 
#                      is used for further analysis.
def preprocess(corpus):
    
    processed = []
    
    # initialising a lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()
    
    for review in corpus:
        
        review = review.lower()      
        tokens = nltk.tokenize.word_tokenize(review)
        review = [lemmatizer.lemmatize(w) for w in tokens]
        review = ' '.join(review)
        
        processed.append(review)
        
    return processed