# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:29:03 2019

@author: Yehuda
"""
import cv2
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance
import seaborn as sns
#mnist data set

#normlize data

#for i in range(m):
#    img=data.iloc[i,:]
#    data.iloc[i,:]= cv2.cvtColor(data.iloc[2,:], cv2.COLOR_BGR2GRAY)
#data_mean=np.array(data.mean(axis=0)).reshape(1,n)
#
#data=np.array(data)
#for i in range(m):
#    data[i,:]=data[i,:]-data_mean
#std=[]
#for i in range(n):
#    std.append((data[:,i]**2).mean())
#for i in range(n):
#    data[:,i]=data[:,i]/std[i]

class TSNE():
    def __init__(self,data):
        self.m,self.n=data.shape
        self.x=data
        self.y=np.random.normal(0,1/(10**4),(self.m,2))
        self.sigma=1*np.ones(self.m)
      
    def euclid_dist(self,v1,v2):
        return (((v1-v2)**2).sum())**(0.5)

    def caculate_pij(self,euclidean_dist,vector_euclidean_dist,sigma):
        sigma=2*(sigma**2)
        t=np.exp(-(euclidean_dist**2)/sigma)
        
        vector_euclidean_dist=-(vector_euclidean_dist**2)/sigma
        return (t/(np.exp(vector_euclidean_dist)).sum())
        
    def caculate_p(self,x_dist):
        m=self.m
        p=np.zeros([m,m])
        for i in range(m):
            for j in range(m):
                if i!=j:
                    if x_dist[i,j]==0:
                        x_dist[i,j]=x_dist[j,i] 
                    p[i,j]=self.caculate_pij(x_dist[i,j],np.delete(x_dist[i,:],i),self.sigma[i])
        return (p+p.T)/(2*self.m)

            
    def caculate_q(self,y_dist):

        t1=(1+y_dist**2)**(-1)
        sum=0
        for k in range(self.m):
            for l in range(self.m):
                if k!=l:
                    if(y_dist[l,k]==0):
                        y_dist[l,k]=y_dist[k,l]
                    t2=(1+y_dist[k,l]**2)**(-1)
                    sum+=t2
                else:
                    t1[k,l]=0
        return t1/sum
#    def cac_sigma(sigma,dist_):
#        perp=lambda p:2**p
        
    
    #update distances
    def caculate_dist(self,a):
        m,n=a.shape
        dist_matrix=np.zeros([m,m])
        for i in range(m):
            for j in range(i,m):
                v1=np.array(a[i,:])
                v2=np.array(a[j,:])
                dist_matrix[i,j]=self.euclid_dist(v1,v2) 
        return dist_matrix      
                
                
    def kullback_leibler_dive(self,p,q,y_dist,i):
         sum=0
         part1=p[i,:]-q[i,:]
         for k in range(self.m):
             sum+=part1[k]*(self.y[i,:]-self.y[k,:])*(1+(y_dist[i,k]**2))**(-1)
         self.y[i,:]+= (1)*4*sum  
    def entropy(self,p_i):
        return -(p_i*np.log2(p_i)).sum()
    def cacu_perplexity(self,x_dist,p,i,target=30):
        preplexity=2**self.entropy(p[i,:])
        while distance.euclidean(preplexity,target)>=1:
            if(preplexity-target)<-1:
                    self.sigma[i]+=10
            else:
                    self.sigma[i]-=10
            p=self.caculate_p(x_dist)
        
        return p
    def los_function(self,p,q):
        sum1=np.float(0)
        sum2=np.float(0)
        for i in range(self.m):
            for j in range(self.m):
                if i!=j:
                    sum2+=p[i,j]*(np.log(p[i,j]/q[i,j]))
            sum1+=sum2
            sum2=0
        return sum1     
# =============================================================================
# =============================================================================
        
        
if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    
    x=(iris.data)
    np.random.shuffle(x)
    
    
    a=TSNE(x[0:50,:])
    
    x_dist=(a.caculate_dist(np.array(a.x)))
    y_dist=(a.caculate_dist(np.array(a.y))) 

    p=a.caculate_p(x_dist)
    
    q=a.caculate_q(y_dist)           
#    for i in range(a.m):
#        p=a.cacu_perplexity(x_dist,p,i)
    print(a.los_function(p,q))
    for i in range(1000):
      
        
        
        for i in range(a.m):
        
            a.kullback_leibler_dive(p,q,y_dist,i)
            
            y_dist=a.caculate_dist(a.y)
            q=a.caculate_q(y_dist) 


    color=['blue']*10+['red']*10
    t=a.y
# plot the result
    vis_x =t[:, 0]
    vis_y =t[:, 1]
    plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
  
    plt.show()