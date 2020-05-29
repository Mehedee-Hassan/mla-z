import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\reinforcement_learning\\"
dataset = pd.read_csv(path+"Ads_CTR_Optimisation.csv")



# Implementing UCB
# at each round n we consider two numbers for each ad i:
# Ni (n) - the number of times the ad i was selected up to roudn n,
# Ri (n) - the sum of rewards of the ad i up to round n.
# stage 2:
# - the average reward of ad i up to round n 
N =10000
d=10
ads_selected =[]   #

numbers_of_selections = [0] * d # a vector containing only 0s of size d 
# at the initial stage the random selection is zero
sums_of_rewards = [0] * d
total_reward = 0
import math

for n in range(0,N):
    
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
#sepecific round and specific version of ad
        #
        if numbers_of_selections[i]> 0:
                
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
                    
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
                
            upper_bound = average_reward  + delta_i
        else:
            upper_bound=1e400
            
        
        if max_upper_bound< upper_bound:
            max_upper_bound = upper_bound
            ad = i
        
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad]+reward 
    total_reward = total_reward + reward
    
    
    
        
         #confidence_interval =  
        
        