# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:46:07 2019

@author: Yehuda
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:13:01 2019

@author: Yehuda
"""
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

df=pd.read_csv("file:///C:/Users/Yehuda/.spyder-py3/wdbc.data",header=None)
# =============================================================================
#Data class shuffle Data and split it to test&train
# =============================================================================
class Data(object):
    def __init__(self, data,splitRatio):
        data=data.sample(frac=1)
        trainsize=int((data.shape[0])*splitRatio)
        self.train=data.iloc[:trainsize,:]
        self.test=data.iloc[trainsize:,:]
class node(object):
    def __init__(self,data,right_n=np.nan,left_n=np.nan,feature_ind=np.nan,value=np.nan,depth=0,leaf_val=np.nan):
        self.right_n=right_n
        self.left_n=left_n
        self.feature_ind=feature_ind
        self.value=value
        self.depth=depth
        self.data=data
        self.leaf_val=leaf_val
# =============================================================================
# decsion tree algoritem given test data & train
# =============================================================================
class DecsionTree():
    def __init__(self, train,test):
        self.train=train
        self.test=test
        
# ****************************************************************************
#main_jini_calc-
# get the training set and sorting each feature.put each sort feature in a list 
# for each item in the list(sort feature) calculate the avrerage between 2 data
#send each average list &column for checking the best jini & value. 
#put in JIniDF and pick the feature with minmum jini crteria
# ****************************************************************************
   
    def main_jini_calc(self,data):
        j=0
        jiniDf=pd.DataFrame([],columns=['jini','mean_val','ind'],index=list(np.arange(0,data.shape[1]-2)))
        
        for i in list(data.columns[2::]):
            sortDf=data.sort_values(by=i)
            average=[data[i][j:j+2].mean() for j in range(0,len(data[i]))]
            row=self.best_jini_feature(sortDf,average,i)
            if (len(row)==1):
                jiniDf.iloc[j,:]=[row.iloc[0,0],row.iloc[0,1],i]
            else:
                 jiniDf.iloc[j,:]=[row[0],row[1],i]
            j+=1
            
        
        result=jiniDf[jiniDf['jini']==(jiniDf['jini'].min())]
        return (result.iloc[0:1,:]) #put attention!!!
    
# ****************************************************************************
#best_jini_feature
#for each feature calc the best jini
#input=sortDf=data.sort_values(by=1),list of average,i=the num of column
#ouput=the best jini ,value,i=num of column
# ****************************************************************************
        
    def best_jini_feature(self,sortDf,average,i):
        
        col=sortDf[i]
        jini_df=pd.DataFrame([],columns=['jini','mean_val','ind'],index=list(np.arange(0,len(average))))
        j=0
        for value in average:
            r_leaf,l_leaf=self.leaf_dvide(col,sortDf,value)
            jini_average=self.jini_calc_leaf(r_leaf,l_leaf)
            jini_df.iloc[j,:]=[jini_average,value,i]
            j+=1
         
        result=jini_df[jini_df.jini==(jini_df.jini.min())]
      
        return (result.iloc[0:1,:])
# ****************************************************************************
#divide the leaf for right anf left
# ****************************************************************************
    
    def leaf_dvide(self,col,sortDf,value):  
        r_leaf=sortDf[1][col>value].value_counts()  
        l_leaf=sortDf[1][col<value].value_counts()  
        return r_leaf,l_leaf
    
# ****************************************************************************
#jini calc for the right&left leafs 
# ****************************************************************************            
    def jini_calc_leaf(self,r_leaf,l_leaf):
        if(not(r_leaf.empty)):
            calc_r=1-((r_leaf/r_leaf.sum())**2).sum()
        else:
            calc_r=0
        if(not(l_leaf.empty)):
            calc_l=1-((l_leaf/l_leaf.sum())**2).sum()
        else:
            calc_l=0
        m=r_leaf.sum()+l_leaf.sum()
        jini_average=(r_leaf.sum()/m)*calc_r+(l_leaf.sum()/m)*calc_l
        return jini_average

# ****************************************************************************
#split Data given data,index of feature and value
# **************************************************************************** 

    def split_data(self,data,index_f,value):
       
        right=data[data[index_f]>=value]
        left=data[data[index_f]<value]
        return right,left
    
    
    
    def split_node(self,n):
        
        if (n.depth==8):
            pass
        elif(len(n.data[1].unique())==1):
            n.leaf_val=str(n.data.iloc[0,1])
            
        else:
            t=self.main_jini_calc(n.data)
            
            n.feature_ind,n.value=t.iloc[0,2],t.iloc[0,1]
            data_r,data_l=self.split_data(n.data,n.feature_ind,n.value)
            n.right_n=node(data=data_r,depth=n.depth+1)
            n.left_n=node(data=data_l,depth=n.depth+1)   
            self.split_node(n.right_n)
            self.split_node(n.left_n)
            
            
 
    
    def predict(self,row,node):
#        print(node.depth)
        if (node.leaf_val == node.leaf_val):
            return node.leaf_val
        if (row[node.feature_ind]>node.value):
#            print('right')
            return self.predict(row,node.right_n)
        else:
#            print('left')
            return self.predict(row, node.left_n)
        


            
            
         
# =============================================================================
# =============================================================================
        
        
if __name__ == "__main__":
    dat=Data(df,2/10)
    Algo=DecsionTree(dat.train,dat.test)
    
    nod=node(dat.train)
  #  tr=Algo.main_jini_calc(dat.train)
    vo=[]
    Algo.split_node(nod)
    for i in range(len(dat.test)):
        vo.append(Algo.predict(dat.test.iloc[i,:],nod))
    accurecy=(vo==dat.test[1]).value_counts()
    result=int(accurecy[1]/accurecy.sum()*100)