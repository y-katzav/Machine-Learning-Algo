import numpy as np
import matplotlib.pyplot as plt

m=np.random.randint(1,10,size=(10))
x1=np.random.rand(10,1)

#2
#2a
gau_d = np.random.normal(0,1,size=(10,1))
first_array=2*x1+gau_d

#2b
x2=np.random.rand(10,1)
second_array=3*x2+6+gau_d

#2c
x3=np.random.rand(1,20)
third_array=2*x3**2

#3
A=np.random.randint(0,20,size=(4,4))
B=np.random.randint(0,20,size=(4,4))
C=A*B
C1=np.linalg.inv(C)
C2=C.transpose()

#4

nt=x1.transpose()
temp=np.linalg.inv(np.dot(nt,x1))*nt
h=np.dot(temp,first_array)
e=h*x1+gau_d 



#5
t=np.ones(10)
A= np.column_stack((x2,t))
At=A.transpose()
temp1=np.dot(np.linalg.inv(np.dot(At,A)),At)

h1=np.dot(temp1,second_array)
e1=np.dot(h1.transpose(),At).transpose()
plt.plot(x2,second_array,'bs',x2,e1,'r--')


#6
plt.plot(x1,first_array,'bs',x1,e,'r--')

#7
x3t=x3.transpose()
temp2=np.dot(np.linalg.inv(np.dot(x3t,x3)),x3t)
