# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:13:32 2019

@author: Yehuda
"""

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
    
    
        def __init__(self,x,s,p,target,f=[3,3],l=6,mini_batch=30):
            self.filter=np.random.normal(0,0.03,[f[0],f[1],l])#size df filter ntw
            self.w_connect=np.random.normal(0,0.05,[10,384])
            self.w=x.shape
            self.s=s
            self.p=p
            self.x=self.normlize(x)
            self.l=l#number of filtter
            self.b=mini_batch
            self.target=target
        def load_filter(self):
            pass
        def forward(self,p,s,b,start):#b=mini-batch
            f=self.filter
            l=self.l
#            size=int((8-f.shape[1]+2*p)/s)+1
            f1_img=np.zeros([8,8,self.l,b])
            for i in range(b):
                for j in range(l):
                    sam=np.pad(self.x[start+i,:].reshape(8,8),((1,1),(1,1)),'edge')#sample
                    f1_img[:,:,j,i]=self.conv(sam,p,s,f[:,:,j],sigmoid)
            return f1_img

        def conv(self,sample,p,s,f,function):
            x=sample
            m=f.shape[1]
            step=int((x.shape[1]-m)/s)+1
            f_img=np.zeros([step,step])
            for i in range(0,step,s):
                for j in range(0,step,s):
                    t=x[i:m+(i),j:m+(j)]
                    f_img[i,j]=function(np.multiply(f,t).sum())
            return f_img
        def f_connect(self,f_img,l):
            result=np.zeros([self.b,l])
            for i in range(self.b):
                flat_img=f_imgs[:,:,:,i].flatten()
                w=self.w_connect
                result[i,:]=self.soft_max(np.dot(w,flat_img))

            return result
        def soft_max(self,z):
            m=z.shape[0]
            
            result=np.zeros([m])
            for i in range(m):
                result[i]=np.exp(z[i])/(np.exp(z)).sum()
                
            return result
        def der_softmax(self,z):
            return self.soft_max(z)*(1-self.soft_max(z))
        def backprop(self,sample,target,yp,l1):#z=w_connect *layer,w2=w_connect
            sample=np.pad(sample.reshape(8,8),((1,1),(1,1)),'edge')
            target=self.traget2vec(target)
            
            error2=target-yp
            l1=l1.flatten()
            z=np.dot(self.w_connect,l1)
            delta2=np.multiply(error2,self.der_softmax(z))#len 10
            
            error1=np.dot(self.w_connect.T,delta2)#len384
            
            z=np.zeros([8,8,self.l])
            for i in range(self.l):
                z[:,:,i]=self.conv(sample,1,1,self.filter[:,:,i],der_sigmoid)
            delta1=np.multiply(error1,z.flatten())
            delta1=delta1.reshape(8,8,6)
            return (delta2.reshape(10,1)*l1.reshape(1,384)),self.new_w1(sample,delta1)
        def traget2vec(self,target,l=10):
            temp=np.zeros([l])
            temp[target]=1
            return temp
            
        def new_w1(self,sample,delta,step=3):
            l=self.l
            m=delta.shape[1]
            new_w=np.zeros([step,step,l])
            for k in range(l):
                f=delta[:,:,k]
                for i in range(step):
                    for j in range(step):
                        t=x[i:m+(i),j:m+(j)]
                        new_w[i,j,k]=(np.multiply(f,t).sum())
            return new_w
        def los_function(self,yp):
            c=np.sqrt(((self.target-yp)**2).sum())
            return (0.5)*c
        def normlize(self,x):
            x-=x.mean(axis=0)
            for i in range(1,x.shape[1]):
                if (i!=39) and (i!=32):
                    x[:,i]/=((x[:,i]**2).mean())**(0.5)
            return (x)   
# =============================================================================
# =============================================================================
if __name__ == "__main__":
    digits = load_digits()
    x=digits.data
    target=digits.target
# =============================================================================
#    trainig -num of training=1440 examples ,epochs=48,mini batchs=30
    num_train=1770
    a=cnn(x,1,1,target)
    for start in range(0,num_train,30):
        

        print(start)
        for d in range(5):
            f_imgs=a.forward(1,1,30,start)
            ful_conn=a.f_connect(f_imgs,10)
            sum1=0
            sum2=0
            for i in range(a.b):
                q2,q1=a.backprop(x[start+i,:],target[start+i],ful_conn[i,:],f_imgs[:,:,:,i])
                sum1+=q1
                sum2+=q2
            
            
            a.filter+=(sum1/a.b)
            a.w_connect+=(sum2/a.b)

# =============================================================================
#            prdiction area 
            
    yp=np.argmax(ful_conn,axis=1)
    dd=target[0:0+30]
    az=(yp==dd)
    az=np.multiply(az,1)
    np.count_nonzero(az)
    
    num_test=1770
    yp1=np.zeros(num_test)
    
    for u in range(0,num_test,30):
        f_imgs=a.forward(1,1,30,u)
        ful_conn=a.f_connect(f_imgs,10)
        yp1[u:u+30]=np.argmax(ful_conn,axis=1)
    yp1=yp1.astype(int)
    mty=yp1[0:num_test]==target[0:num_test]
    mty=np.multiply(mty,1)
    
    acuurecy=(np.count_nonzero(mty)/num_test)*100
    
    
#    op=a.x
#    for i in range(1,op.shape[1]):
#        if (i!=39) and (i!=32):
#            op[:,i]/=((op[:,i]**2).mean())**(0.5)
#    
#    i=np.isnan(x)
    
    
# =============================================================================



#        
#    def pad(self,x,p):
#        m,n=x.shape
#        pad=np.zeros([m+2*p,m+2*p])
#        pad[p:n+p,p:n+p]=x   
#        return pad
