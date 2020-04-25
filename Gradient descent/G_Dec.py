import numpy as np
import matplotlib.pyplot as plt
"""
THIS IS "TIOTA" PAPERS

X=np.array([[31,22],[22,21],[40,37],[26,25]])
y=np.array([2,3,8,12])
y=y.reshape(4,1)
tetah=np.zeros([2,1])
cost_f=0.5*(np.sum(((X@tetah)-y)**2))
x1=X[::,0].reshape(1,4)
x2=X[::,1].reshape(1,4)
for i in range(5):
    temp1=tetah[0]-np.sum(((X@tetah)-y)@x1)
    temp2=tetah[1]-np.sum(((X@tetah)-y)@x2)
    tetah[0]=temp1
    tetah[1]=temp2
    
cost_f=0.5*(np.sum(((X@tetah)-y)**2))
result=X@tetah

#1

X=np.array([[31,22],[22,21],[40,37],[26,25]])
Xt=X.transpose()
y=np.array([2,3,8,12]).reshape(4,1)

b=np.linalg.inv((Xt@X))@Xt@y
result=X@b
#2
x3=(X[::,0]-X[::,1]).reshape(4,1)
X= np.column_stack((X,x3))
Xt=X.transpose()
b=np.linalg.inv((Xt@X))@Xt@y
result1=X@b

X= np.column_stack((X,np.ones([4,1])))
Xt=X.transpose()
b=np.linalg.inv((Xt@X))@Xt@y
result2=X@b


class Linear():
    X=np.array([[31,22],[22,21],[40,37],[26,25]])
    Xt=X.transpose()
    y=np.array([2,3,8,12]).reshape(4,1)
    
    def cal(self,v):
        v=np.array(0)
        if v.any!=0:
            self.X= np.column_stack((self.X,v))
        return np.linalg.inv((self.Xt@self.X))@self.Xt@self.y
    def get_X(self):
        return self.X
A=Linear()
b1=A.cal(0)
result45=(A.get_X())@b1
#2
temp=tetah
i=(X1@tetah-labels).transpose()@(X1[::,1])
for i in range(1000):
    for j in range(3):
        temp[j]=tetah[j]-((0.1)*(((X1@tetah)-labels).transpose()@X1[::,j]))/3
    tetah=temp
cost_f=0.5*(np.sum(((X1@tetah)-labels)**2))

U=np.linalg.inv((X1t@X1))@X1t@labels

Gradient DECENDING
"""



##Linear regression:
#A
X=np.array([[31,22],[22,21],[40,37],[26,25]])
Xt=X.transpose()
y=np.array([2,3,8,12]).reshape(4,1)

b=np.linalg.inv((Xt@X))@Xt@y

#B
x3=(X[::,0]-X[::,1]).reshape(4,1)
X= np.column_stack((X,x3))
Xt=X.transpose()
b=np.linalg.inv((Xt@X))@Xt@y

#D
X= np.column_stack((X,np.ones([4,1])))
Xt=X.transpose()
b=np.linalg.inv((Xt@X))@Xt@y
result2=X@b

##Gradient DECENDING
#Stochastic gradient descent
features=np.array([[0],[1],[2]])
labels=np.array([[1],[3],[7]])
X1=np.array(np.column_stack(([1,1,1],features,features**2)))
tetah=np.array([[2],[2],[0]])
temp=np.ones([3,1])
print(((X1@tetah)-labels).reshape(1,3)@X1[::,0])

for i in range(1000):
    
    for j in range(3):
        temp[j]=tetah[j]-((0.1)*(((X1@tetah)-labels).reshape(1,3)@X1[::,j]))
    tetah=temp

#CMomentum
gama=0.9
v=0
temp=np.zeros([3,1])
for i in range(1000):
    
    for j in range(3):
        v=gama*v+((0.1)*(((X1@tetah)-labels).reshape(1,3)@X1[::,j]))
        temp[j]=tetah[j]-v
    tetah=temp
 
#DNesterov accelerated 
gama=0.9
v=0
temp=np.zeros([3,1])
for i in range(2):
    
    for j in range(3):
        v=gama*v+((0.01)*(((X1@tetah)-labels).reshape(1,3)@X1[::,j]))
        temp_t=tetah[j]-v
        temp[j]=temp_t-gama*v
    tetah=temp  

