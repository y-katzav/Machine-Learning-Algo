# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:19:20 2019

@author: Yehuda
"""
import sympy as sym
import numpy as np
#
#def sigmoid(x):
#    return 1/(1+np.exp(-x))


fn=lambda x: 1/(1+np.exp(-x))

der_fn=lambda x:fn(x)*(1-fn(x))

error=lambda yp,y:(1/2)*(yp-y)**2 

w=np.random.rand(3,1)

x_input=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

y=np.array([[0],[0],[1],[1]])

def back_prop(x_input,w,y,alpha=0.1):
    z=np.dot(x_input,w)
    
    y_predict=fn(z)
    error=y-y_predict
    sigmoid_derivative=der_fn(z)
    temp=sigmoid_derivative*error
    
    gradient=np.dot(x_input.T,temp)
    
    return w+alpha*gradient

for i in range(1000):
    w=back_prop(x_input,w,y)
    
temp=np.dot(x_input,w)
np.round((fn(temp)))


#add another layer
w1=np.random.rand(3,3)
#second Layer
a2=np.dot(w1,x_input.T)

#wighets output layer
w2=np.random.rand(3,1)
#y predict
yp1=np.dot(a2.T,w2)
#y tareget
y=np.array([[0],[1],[1],[0]])

def backpro(x_input,w1,w2,y,alpha=0.01):
    m,n=x_input.shape
#    l1=4x3 -each row is a layer         in w1 every column is a wighet
    l1=fn(np.dot(x_input,w1))
#    l1=4x3 eech row is a layer per input
   
#     l2=4X1 each sample/layer to single neuron
    l2=fn(np.dot(l1,w2))
#    y_predict/ single neuron per sample
    
    
#    4x1
    l2_error=y-l2
#    4x1*4x1=4x1
    l2_delta=l2_error*der_fn(np.dot(l1,w2))
#    4x3
    l1_error=np.dot(l2_delta,w2.T)
    
#    3x3
    l1_delta=np.zeros([m,n])
    temp=der_fn(np.dot(x_input,w1))
    for i in range(m):
        l1_delta[i,:]=temp[i,:]*l1_error[i,:]
    
    w2+=np.dot(l1.T,l2_delta)
    w1+=np.dot(x_input.T,l1_delta)
    
    return w1,w2
for i in range(1000):
    w1,w2=backpro(x_input,w1,w2,y)
l1=fn(np.dot(x_input,w1))
y_predict=fn(np.dot(l1,w2))
np.round(y_predict)
#def backp(x_input,w1,w2,y,alpha=0.01):
##     x_input row vector
#    
#    l1=fn(np.dot(x_input,w1))
##    y predict w2column vector
#    l2=fn(np.dot(l1,w2))
#    l2_error=y-l2
#    
#    l2_delta=l2_error*der_fn(np.dot(x_input,w1))
#    
#    l1_error=np.dot(l2_delta,w2)
#    
#    l1_delta=der_fn(np.dot(x_input,w1))*l1_error
#    
#    w1+=x_input*l1_delta
#    w2+=l1*l2_delta
#    
#    return w1,w2
#
#w1,w2=backp(x_input,w1,w2,y1)
#    
#    