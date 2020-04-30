# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:18:48 2019

@author: Yehuda
"""

import numpy as np

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

sigmoid=lambda x: 1/(1+np.exp(-x))

der_sigmoid=lambda x:sigmoid(x)*(1-sigmoid(x))


# in this algoritm my starting point is the input is nxn
#the paddinfg is also1, as well the stride
class cnn():
    
    
        def __init__(self,x,s,p,f=[3,3],l=6,mini_batch=30):
            self.filter=np.random.rand(f[0],f[1],l)#size df filter ntw
            self.w_connect=np.random.rand(1,384,10)
            self.w=x.shape
            self.s=s
            self.p=p
            self.x=x
            self.l=l
            self.b=mini_batch
        def load_filter(self):
            pass
        def forward(self,p,s,b):#b=mini-batch
            f=self.filter
            l=self.l
#            size=int((8-f.shape[1]+2*p)/s)+1
            f1_img=np.zeros([8,8,self.l,b])
            for i in range(b):
                for j in range(l):
                    sam=self.pad(self.x[i,:].reshape(8,8),1)#sample
                    f1_img[:,:,j,i]=self.conv(sam,p,s,f[:,:,j])
            return f1_img
        def pad(self,x,p):
            m,n=x.shape
            pad=np.zeros([m+2*p,m+2*p])
            pad[p:n+p,p:n+p]=x   
            return pad
        def conv(self,sample,p,s,f):
            x=sample
            m=f.shape[1]
            step=int((x.shape[1]-m)/s)+1
            f_img=np.zeros([step,step])
            for i in range(0,step,s):
                for j in range(0,step,s):
                    t=x[i:m+(i),j:m+(j)]
                    
                    sum=0
                    for k in range(m):
                        for l in range(m):
                            sum+=t[k,l]*f[k,l]
                    f_img[i,j]=sigmoid(sum)
            return f_img
        def f_connect(self,f_img,l):
            result=np.zeros([self.b,l])
            for i in range(self.b):
                flat_img=f_imgs[:,:,:,i].flatten()
                w=self.w_connect
                
                for j in range(l):
                    result[i,j]=np.dot(w[:,:,j],flat_img)
            for i in range(self.b):
                result[i,:]=self.soft_max(result[i,:])
            return result
        def soft_max(self,z):
            m=z.shape[0]
            
            result=np.zeros([m])
            for i in range(m):
                result[i]=np.exp(z[i])/(np.exp(z)).sum()
                print(result[i])
            return result
        def backprop(self,target,yp,z,w1,w2):
            error=yp-target
            der=der_sigmoid(z)
# =============================================================================
# =============================================================================
if __name__ == "__main__":
    digits = load_digits()
    x=digits.data
    target=digits.target
#    np.random.shuffle(x)
#    plt.gray() 
#    img=x[0].reshape(8,8)
#    plt.imshow(img)
    
    a=cnn(x,1,1)
    f_imgs=a.forward(1,1,30)
    ful_conn=a.f_connect(f_imgs,10)
    a.filter