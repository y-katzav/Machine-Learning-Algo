# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:56:20 2019

@author: Yehuda
"""

# Python program to read  
# image using matplotlib 
  
from scipy import misc,linalg
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
import os
from sklearn.decomposition import PCA

# Read Images 
data=np.zeros([180*200,153])

i=0
path=("C:\\Users\Yehuda\Desktop\Machine learning\eigenfaces")
for p in os.listdir(path):
    new_path=os.path.join(path,p)
    for k in os.listdir(new_path):
        temp=os.path.join(new_path,k)
        img= os.listdir(temp)
        if img[0].endswith('.jpg'):
            face = misc.imread(os.path.join(temp,img[0]),flatten=True)
        else:
            face = misc.imread(os.path.join(temp,img[2]),flatten=True)
        data[:,i]=face.reshape(36000)
        i+=1
#        temp=os.path.join(new_path,k)
#        t_l=os.listdir(temp)
#        
data = data.T
#PCA by skicit_learn
pca = PCA(n_components=30, copy=True, svd_solver='randomized')
projected_data=pca.fit(data).components_
#plot the projectes data
plt.imshow(projected_data[0,:].reshape(200,180))


#my PCA
#centerd the data
miu=np.mean(data,axis=0)
#for fum plotting the miu
plt.imshow(miu.reshape(200,180))
x=data-miu
m,n=data.shape
sigma=[((x[:,i]**2).sum())/m for i in range(m)]
for i in range(m):
    x[:,i]=x[:,i]/(sigma[i]**(0.5))
#covraince_matrix=np.dot(x.T,x)/m

u,s,vh=np.linalg.svd(x,full_matrices=False)
#my projected data
temp=np.zeros([30,200*180])
temp[0:30,0:30]=np.diag(s[0:30])
p_data=np.dot(u,temp)
#plot projected data

plt.imshow(p_data[0,:].reshape(200,180))


#second try my svd using wikpedia
# link https://en.wikipedia.org/wiki/Eigenface

#ach row contains one mean-subtracted image
x1=data-miu
#153x153
t1=np.dot(x1,x1.T)
#eigen vectors of t1
eigen_value,eigen_vectors=np.linalg.eig(t1)
#for getting "covarience matrix" with trick in the link uplow
data_eigen_vectors=np.dot(x1.T,eigen_vectors)

#my coveraiance matrix (need more to understand the next 2 steps)
result=np.dot(x1,data_eigen_vectors)
#projecting data
projecting_data=np.dot(x1.T,result)


#plot the result
plt.imshow(projecting_data[:,45].reshape(200,180))
"""
pca = PCA(n_components=30, copy=True, svd_solver='randomized')
eigenfaces = pca.fit(data)
plt.imshow(T[78,:].reshape(200,180))
plt.imshow(eigenfaces[1].reshape((200,180)))
u=np.dot(data,data.T)/152
u_,ei=linalg.eig(u)
ei=ei[:,0:30]
sigma=np.zeros([153,200*180])
sigma[0:153,0:153]=(np.diag(u_))
sigma=sigma[0:30,:]
T=np.dot(ei,sigma)

"""