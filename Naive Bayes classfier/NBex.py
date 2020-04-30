# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:39:57 2019

@author: Yehuda
"""

import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt


'''a Data class
  Attributes:
  train-data for the train set
  test-data for test part
'''
class Data(object):
    def __init__(self, data,splitRatio):
        data=data.sample(frac=1)
        trainsize=int((data.shape[0])*splitRatio)
        self.train=data.iloc[:trainsize,:]
        self.test=data.iloc[trainsize:,:]
    
class NaiveBase(object):
    def __init__(self,train,test):
        self.train=train
        self.test=test
        
    #get the train data, and for each feature calculate 'mean','std' groupby class
    #B Number of times the class appears
    def Summeraize(self):
        colm=self.train
        A=colm.groupby(colm.columns[-1]).agg(['mean','std'])
        B=colm[colm.columns[-1]].value_counts()
        summery=pd.concat([A,B],axis=1)
        return summery
    #the gaussian fuction     
    def Gauss(self,v,me,st):
        exponent=np.exp(-(np.power(v-me,2))/(2*(st**2)))
        gaussian=(1/np.sqrt(2*np.pi*(st**2)))*exponent
        return gaussian
    '''take the test set, mean and std for each feauter group by class.for
       each faeture in test set it in the gassiuan function group by class(A for class 0
       and B for class 1) ,last caculate
       
       the probility for each class
    '''
    def predict(self):
        tdata=self.test
        summery=self.Summeraize()
        A=np.ones([len(tdata),tdata.shape[1]])
        B=np.ones([len(tdata),tdata.shape[1]])
        for i in range(tdata.shape[1]-1):
            A[:,i]=self.Gauss(tdata[i],summery.iloc[0,i],summery.iloc[0,i+1])
            B[:,i]=self.Gauss(tdata[i],summery.iloc[1,i],summery.iloc[1,i+1])
        temp=summery.iloc[:,-1]/len(self.train)
        A[:,-1]=temp[0]
        B[:,-1]=temp[1]
        #A The probilaity of being calss 0 ,B the probiaity of being class 1
        #product the feartures for each instance
        A1=np.ones([len(A),1])
        B1=np.ones([len(A),1])
        
        for i in range(len(A)):
            A1[i]=np.product(A[i,:])
            B1[i]=np.product(B[i,:])
        C=(A1<B1)
        
        return C
        
    
        
    def Accuracy(self):
        A=self.predict()
        A=[int(a) for a in A]
        temp=A==self.test.iloc[:,-1]
        return int((temp.value_counts()[1]/len(self.test))*100) 
    
    
# =============================================================================
## =============================================================================
        
        
if __name__ == "__main__":        
    
    data=pd.read_csv("pima-indians-diabetes.csv",header=None)
    d=Data(data,2/3)
    t=NaiveBase(d.train,d.test)
  
    x=t.Summeraize()
    A=(t.predict())
    
    C=t.Accuracy()
    
 
