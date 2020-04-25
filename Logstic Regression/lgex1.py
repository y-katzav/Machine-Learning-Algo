# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:40:46 2019

@author: Yehuda
"""

import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
X=df.iloc[:,0:2]
y=df['Iris-setosa']=='Iris-setosa'
d1={True:1,False:0}
y=y.map(d1)
d={True:'r',False:'b'}
plt.scatter(X.iloc[:,0],X.iloc[:,1],c=y.map(d))

def sigmoid(z):
    return 1. / (1 + np.exp(-z))

class LogisticRegression(object):
    def __init__(self, featurs, label):
      self.x=featurs
      self.x['inpt']=np.ones((featurs.shape[0], 1))
      self.y=label   #coulmn vector
      self.n=label.size
     
   # def train(self,lr=0.1,tetha=np.zeros(self.n,1)):
       #p_y=sigmoid((self.tetha.transpose()@x))

    def likelihood(self,h,y):
       #y row vector
        return np.mean(-y@np.log(h)-(1-y)@np.log(1-h))
    def gradient(self,h):
        y=self.y
        return np.dot(self.x.transpose(), (h - y)) / y.size
    def get_x(self):
        return self.x
    def split(self):
        return self.x.sample(frac=1)
    
    def fit(self):
        # weights initialization
        theta=np.random.rand(3)
        print(theta) 
        for i in range(100):
            z = theta@self.x.T
            h = sigmoid(z)
            theta-=(0.1) * self.gradient(h)
        print(theta) 
        return theta
    
    def classfier(self,theta):
        d1={True:1,False:0}
        predict=(theta@self.x.transpose())
        
        predict= predict>0.5
        return list(predict.map(d1)).count(1)
# =============================================================================
## =============================================================================
       
        
if __name__ == "__main__":                
        
    Y=df['Iris-setosa']       
    c=LogisticRegression(X,Y)
    w=c.fit()
    q=c.classfier(w)
