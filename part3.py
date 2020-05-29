# -*- coding: utf-8 -*-
# 50_Startups.csv


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values


# all in 
# backard Elimination
# forward selection
# bidirection eliminaiton
# Score Comparison

# backward elimination
# 1 -- select the significance level to stay in the modell 
# 2 -- fit the full model with all possible preditors
# 1 -- consider the predictor with the highst p-value. If P > SL ,go to step 4, otherwise Fn


