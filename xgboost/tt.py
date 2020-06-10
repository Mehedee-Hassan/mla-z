# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:05:21 2020

@author: mehedee
"""
path = "C:\\Users\\mehedee\\Documents\\data\\github\\xgboost\\"
trian_file = 'agaricus.txt.train.txt'
test_file = 'agaricus.txt.test.txt'

import xgboost as xgb
import pandas as pd


data = pd.read_csv(path+trian_file)
# read in data
dtrain = xgb.DMatrix(path+'agaricus.txt.train.txt')
dtest = xgb.DMatrix(path+'agaricus.txt.test.txt')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)