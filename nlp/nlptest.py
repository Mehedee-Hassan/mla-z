import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path = "C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\nlp\\"
dataset = pd.read_csv(path+"Restaurant_Reviews.tsv", delimiter = '\t',quoting=3)

# stemming 
#

import re
import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus =[]

dataset.shape

for i in range(0,1000):
    
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) 
    eview = review.lower()
    
    review = review.split()
    ps = PorterStemmer()
    
    review2 = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review2) 
    
    corpus.append(review)


# Creating bag of words model
    
from sklearn.feature_extraction.text import CountVectorizer

    
cv = CountVectorizer(max_features=1600)
C = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,1].values








 


