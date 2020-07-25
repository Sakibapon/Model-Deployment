# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 01:40:14 2020

@author: Sakib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
dataset = pd.read_csv('hiring.csv')
dataset['experience'].fillna(0,inplace=True)
dataset['test_score(out of 10)'].fillna(0,inplace=True)
y=dataset['salary($)']
def word2int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]
dataset['experience']=dataset['experience'].apply(lambda a:word2int(a))
x=dataset.iloc[:,:3]
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
pickle.dump(lr, open('model.pkl','wb'))