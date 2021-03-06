#%reset -f


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\reinforcement_learning\\"
dataset = pd.read_csv(path+"Ads_CTR_Optimisation.csv")


import random
N =1000
d =10
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward =dataset.values[n,ad]
    total_reward = total_reward + reward
    
    
plt.hist(ads_selected)
plt.title("Histogram of ads selections")

plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()



