# -*- coding: utf-8 -*-


import numpy as np
from numpy import random,sum,dot
from itertools import combinations
import matplotlib.pyplot as plt

from numpy.linalg import inv

import pandas  as pd
import matplotlib.pyplot as plt

class GMM():
    def __init__(self,k):
        
        self.k=k
        

    def est_step(self,x,u,c,w):
        n,d=x.shape
        

        p=np.zeros([n,self.k])
       
        for j in range(self.k):
            for i in range(n):
                t=w[j]*self.n(x[i,:],u[j],c[j])
                p[i,j]=t/sum([w[k]*self.n(x[i,:],u[k],c[j]) for k in range(self.k)])
#                p[i,j] is p(c/xi)=p(c)*p(xi/c)/p(xi)

        
        result=[]
        for i in range(n):
            result.append(np.argmax(p[i,:])) 
        np.array(result)
        return np.reshape(result,[n,1]),p
    def max_step(self,x,p,result):
        mc=[sum(p[:,i]) for i in range(self.k)]
        w=np.array(mc)/sum(mc)
        u=np.multiply(1/np.array(mc),dot(p.T,x).T)
        cov_matrix=[]
        for i in range(self.k):
            b=np.multiply(p[:,i],(x-u.T[i,:]).T)
            cov_matrix.append((1/mc[i])*dot(b,(x-u.T[i,:])))
        
        return u.T,cov_matrix,w
    def n(self,xi,u,c):
        pa=1/np.sqrt(((2*np.pi)**self.k)*np.linalg.det(sigma))
        gaussian= pa*np.exp(-1/2*(dot(dot((xi-u),inv(c)),(xi-u))))
        return gaussian
    












# =============================================================================
# =============================================================================
if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    
    x=(iris.data)
    np.random.shuffle(x)
    sigma=np.cov(x.T) 
    a=GMM(3)
    w = [1/3] * 3
    u=x[np.random.choice(150, 3, False), :]
    c= [np.cov(x.T)]* 3
    for i in range(1000):
        result,p=a.est_step(x,u,c,w)
        u,c,w=a.max_step(x,p,result)
    ui=pd.DataFrame( np.concatenate((x,result),axis=1))
   

    df_list=[(ui.loc[ui[4]==i]).drop(4,axis=1) for i in range(3)]

    xi=np.array(df_list[0])
    y=np.array(df_list[1])
    z=np.array(df_list[2])
    indx=list(combinations(range(4),2))
    while(indx):
        i=indx[0]
        plt.plot(xi[:,i[0]],xi[:,i[1]],'ro')
        plt.plot(y[:,i[0]],y[:,i[1]],'bo')
        plt.plot(z[:,i[0]],z[:,i[1]],'yo')
        plt.show()
        indx.remove(i)
    
    
    
