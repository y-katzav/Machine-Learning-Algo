# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:00:27 2019

@author: Yehuda
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:23:01 2019

@author: Yehuda
"""

import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from itertools import combinations 

class DBSCAN():
    clids={'UNCLASSIFIED','CLASSIFIED','NOISE'}
    
    def __init__(self,set_of_points,eps,min_pts):
        self.set_of_points=set_of_points
        self.eps=eps
        self.min_pts=min_pts
    
    def nextId(self,cluster_id):
        self.clids.add(cluster_id)
    def expand_cluster (self,class_points,point,clid,eps,min_pts):
        f=DIstanceFunc()
        seeds=class_points.region_query ((class_points.set_of_points),f.euclidean_distance,point.value,eps)
        if(len(seeds)<min_pts):
            class_points.change_clid(point,'NOISE')
            return False
        else:
            class_points.change_clids(seeds,clid)
            seeds.remove(point)
            while(seeds):
                current_p=seeds[0]
                result=class_points.region_query((class_points.set_of_points),f.euclidean_distance,current_p.value,eps)
                if (len(result)>=min_pts):
                    for i in range(len(result)):
                        result_p=result[i]
                        if (result_p in {'UNCLASSIFIED','NOISE'}):
                            if (result_p.clid=='UNCLASSIFIED'):
                                seeds.append(result_p)#end if 
                            class_points.change_clid(result_p,clid) 
                            #end if UNCLASSIFIED or NOISE
                        #end for
                    #end if result.size >= MinPts
                seeds.remove(current_p)
                #end while
            return True
                

            
class SetOfPoints():
    def __init__(self,set_of_points,eps,min_pts):
            self.set_of_points=[]
            for p in set_of_points:
                p=Point(p)
                self.set_of_points.append(p)
            self.eps=eps
            self.min_pts=min_pts
    def region_query (self,data_base,dist_fun,point,eps):
        result=[]
        for p in data_base:
            if (dist_fun(p.value,point)<=eps):
                result.append(p)
        return result
    def change_clid(self,point,clid) :
        point.clid=clid
    def change_clids (self,seeds,clid):
        for p in seeds:
            p.clid=clid
    def get_point(self,i):
        return self.set_of_points[i]
class Point():
    def __init__(self,point):
        self.value=point
        self.clid='UNCLASSIFIED'
class DIstanceFunc():
    def euclidean_distance(self,point_a,point_b):
        return ((point_a-point_b)**2).sum()
# =============================================================================
# =============================================================================
if __name__ == "__main__":
    df=pd.read_csv("file:///C:/Users/Yehuda/Desktop/iris.data",header=None)
    df=df.sample(frac=1)
    df=df.drop(4,axis=1)
    eps=4
    min_pts=3
    data=[np.array(df.iloc[i,:]) for i in range(len(df))]
    dbscan=DBSCAN(data,eps,min_pts) 
    a=SetOfPoints(data,eps,min_pts)
#running the dbscan algoritm   
    cluster_id=0
    for i in range(len(a.set_of_points)):
        point=a.get_point(i)
        if (point.clid=='UNCLASSIFIED'):
            if dbscan.expand_cluster(a,point,cluster_id,eps,min_pts):
                cluster_id+=1
#let's see resuts
    x=[]
    y=[]
    z=[]
    NOISE=[]
    for p in a.set_of_points:
        print(p.value,p.clid)
        if(p.clid==0):
            x.append(p.value)
        elif(p.clid==1):
            y.append(p.value)
        elif(p.clid==2):
            z.append(p.value)
        else:
            NOISE.append(p.value)
    x=np.column_stack(x).T
    y=np.column_stack(y).T
    z=np.column_stack(z).T
    indx=list(combinations(range(4),2))
    while(indx):
        i=indx[0]
        plt.plot(x[:,i[0]],x[:,i[1]],'ro')
        plt.plot(y[:,i[0]],y[:,i[1]],'bo')
        plt.plot(z[:,i[0]],z[:,i[1]],'yo')
        plt.show()
        indx.remove(i)
    comb = list(combinations(range(4), 2))