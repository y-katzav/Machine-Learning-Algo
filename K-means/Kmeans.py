# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:21:37 2019

@author: Yehuda
"""
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

# =============================================================================
#Kmeans Algoritam instance attrbiuts data,k,
# =============================================================================

class Kmeans():
        def __init__(self, data,k):
            self.data=data
            self.k=k
            self.means=[tuple(data.sample(k).iloc[i,:]) for i in range(k)]
            self.clusters={i:[] for i in self.means}
        def cluster_points(self,means):
            
            for i in range(len(self.data)):
                dit={k:self.calc_distance(data.iloc[i,:],k) for k in means}
                m=min(dit.items(), key=lambda x: x[1]) 
                
                self.clusters[m[0]].append(np.array(data.iloc[i,:]))
            
                
        def  reevaluate_centers(self):
            old_clusters=list(self.clusters.keys())
            i=0
            for x in self.clusters:
                a=np.array(self.clusters[x])
                
                self.clusters[tuple(np.mean(a,axis=0))]=self.clusters.pop(x)
                i+=1
            self.means=list(self.clusters.keys())   
            return old_clusters   
        def calc_distance(self,point,value):
            point=np.array(point)
            value=np.array(value)
            return np.sqrt(((point-value)**2).sum())
        
        def has_converged(self,old_clusters,clusters):
            test=[]
            old_clusters=np.array(old_clusters)
            clusters=np.array(clusters)
            
            for i in range(self.k):
                test.append(((old_clusters[i,:]-clusters[i,:])**2).sum()<00.1)
            return test
        
# =============================================================================
# =============================================================================
        
        
if __name__ == "__main__":
    data1 = load_iris()
    df=pd.read_csv("file:///C:/Users/Yehuda/Desktop/Machine learning/תרגילים/iris.data",header=None)
    data=df.sample(frac=1)
    data=data.drop(4,axis=1)
    a=Kmeans(data,3)
    a.cluster_points(a.means)
    b=a.clusters
    old_clusters= a.reevaluate_centers()
    a.cluster_points(a.means)
    tr=(a.has_converged(old_clusters,a.means))
    while(not(np.array(tr).all())):
        old_clusters= a.reevaluate_centers()
        a.cluster_points(a.means)
        tr=(a.has_converged(old_clusters,a.means))
        