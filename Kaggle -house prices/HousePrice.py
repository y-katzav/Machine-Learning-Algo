import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from itertools import combinations 
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor



#RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
#           max_features='auto', max_leaf_nodes=None,
#           min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=1, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
#           oob_score=False, random_state=0, verbose=0, warm_start=False)

#>>> print(regr.feature_importances_)
#[0.18146984 0.81473937 0.00145312 0.00233767]
#>>> print(regr.predict([[0, 0, 0, 0]]))
#[-8.32987858]
### ========================================================
#LOAD DATA AND MAKE IT EASIER FOR NAVGIATION
df=pd.read_csv("train.csv")
df=df.sample(frac=1)
df_columns=list(df.columns)
d_c={i:df_columns[i] for i in range(df.shape[1])}



p=list(combinations(df.columns,2))

## ======================================================================
#      ######      VIZUALIZATION tools ######

## ========================================================
sns.boxplot(x = d_c[7], y = d_c[80], data = df) 
ax=plt.gca()
ax.set_ylim([0,800000])

## ========================================================
sns.distplot(df[d_c[80]])

## ========================================================
plt.scatter(x=df[d_c[43]],y=df[d_c[80]])   #38

ax=plt.gca()
ax.set_xlim([0,2500])
ax.set_ylim([0,800000])
## ========================================================
def numerical_compare(i,k,j):
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(df[d_c[i]],df[d_c[j]],'ro',label=d_c[i])
    ax=plt.gca()
    ax.set_xlim([0,3000])
    ax.set_ylim([0,800000])
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(df[d_c[k]],df[d_c[j]],'bs',label=d_c[k])
    ax=plt.gca()
    ax.set_xlim([0,2000])
    ax.set_ylim([0,800000])
    plt.legend(loc=2)
numerical_compare(43,44,80)
## ========================================================
def numercial_view(i,j):
      plt.plot(df[d_c[i]],df[d_c[j]],'bo',label=d_c[i])
      ax=plt.gca()
      ax.set_xlim([1000,12000])
      ax.set_ylim([0,800000])
numercial_view(4,80)



## ========================================================
#descirbe
def my_describe(i):
    return df[d_c[i]].describe()
my_describe(4)


## ======================================================================
#  end vziualzation tools, start diffrent methods for picking columns ######
#handling with null values

fg=df.isnull().sum()
fg=fg.sort_values(ascending=False)
fg=fg[fg!=0]
#chacking about the pool
pool=df[df[d_c[72]].isnull()==False]
pool.SalePrice.mean()
diffrence=pool.SalePrice.mean()-df.SalePrice.mean()
df[df.SalePrice>pool.SalePrice.mean()]
#

tr=list(df.groupby(d_c[40]).count().index)
tr1={tr[0]:5,tr[2]:4,tr[4]:3,tr[1]:2,tr[3]:1}
df[d_c[40]]=df[d_c[40]].map(tr1)
df[d_c[40]].fillna(0)

df_cat=df.select_dtypes(include=['object'])
#"HouseP - גיליון1.csv" is a file i try to understand the significat of each column
my_excel=pd.read_csv("HouseP - גיליון1.csv")
temp=[]
for i in list(my_excel['Variable']):
    temp.append(i[:-2])
my_excel['Variable']=temp


temp=my_excel.loc[my_excel['Conclusion']=='H']
my_excel.loc[my_excel['Conclusion']=='H/M']
h=df[temp['Variable']]
h=pd.concat([h,pd.DataFrame(df['SalePrice'])],axis=1)

#mapping h Data frame transfer from catgotrical to numeric
tr2=list(df.groupby(d_c[16]).count().index)
#mapping HeatingQC
h[d_c[40]]=h[d_c[40]].map(tr1)
#""""" ExterQual
h[d_c[27]]=h[d_c[27]].map(tr1)
#"""""" BsmtQual
h[d_c[30]]=h[d_c[30]].map(tr1)
h[d_c[30]]=h[d_c[30]].fillna(0)
#"""""" PoolQC
h[d_c[72]]=h[d_c[72]].map(tr1)
h[d_c[72]]=h[d_c[72]].fillna(0)

#
h.select_dtypes(include=['object'])
# mapping CentralAir
h[d_c[41]]=pd.get_dummies(h[d_c[41]])
#mapping  'HouseStyle'
er=list(h.groupby(d_c[16]).SalePrice.mean().sort_values().index)
aw={er[0]:0,er[1]:1,er[2]:2,er[3]:3,er[4]:4,er[5]:5,er[6]:6,er[7]:7}
h[d_c[16]]=h[d_c[16]].map(aw)

#mapping 'GarageFinish'
tr3=list(df.groupby(d_c[60]).count().index)
dc_tr3={tr3[0]:4,tr3[1]:3,tr3[2]:2}
h[d_c[60]]=h[d_c[60]].fillna(1)
h[d_c[60]]=h[d_c[60]].map(dc_tr3)

#mapping Electrical
er1=list(h.groupby(d_c[42]).SalePrice.mean().sort_values().index)
dc_er1={er1[0]:1,er1[1]:2,er1[2]:3,er1[3]:4,er1[4]:5}
h[d_c[42]]=h[d_c[42]].map(dc_er1)
h[d_c[42]]=h[d_c[42]].fillna(0)
## ========================================================
df[df[d_c[59]].isnull()==True]

#try to predict
y=h.SalePrice
h=h.drop(columns=d_c[72])
regr = RandomForestRegressor()
regr.fit(h,y)  



h.isnull().sum()
yp=pd.read_csv("test.csv")
yp=yp.Id
y_predict=pd.DataFrame(regr.predict(test_df))
test_df=pd.concat([yp,y_predict],axis=1)

test_df=pd.read_csv("test.csv")
test_df=test_df[h.columns]

test_df.isnull().sum()
test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(0)


test_df.rename(columns={0:'SalePrice'},inplace=True)
test_df.to_csv("yPredict.csv",index=False)


#mapping h Data frame transfer from catgotrical to numeric
tr2=list(df.groupby(d_c[16]).count().index)
#mapping HeatingQC
test_df[d_c[40]]=test_df[d_c[40]].map(tr1)
#""""" ExterQual
test_df[d_c[27]]=test_df[d_c[27]].map(tr1)
#"""""" BsmtQual
test_df[d_c[30]]=test_df[d_c[30]].map(tr1)
test_df[d_c[30]]=test_df[d_c[30]].fillna(0)
#"""""" PoolQC

#
h.select_dtypes(include=['object'])
# mapping CentralAir
test_df[d_c[41]]=pd.get_dummies(test_df[d_c[41]])
#mapping  'HouseStyle'
er=list(h.groupby(d_c[16]).SalePrice.mean().sort_values().index)
aw={er[0]:0,er[1]:1,er[2]:2,er[3]:3,er[4]:4,er[5]:5,er[6]:6,er[7]:7}
test_df[d_c[16]]=test_df[d_c[16]].map(aw)

#mapping 'GarageFinish'
tr3=list(df.groupby(d_c[60]).count().index)
dc_tr3={tr3[0]:4,tr3[1]:3,tr3[2]:2}
test_df[d_c[60]]=test_df[d_c[60]].fillna(1)
test_df[d_c[60]]=test_df[d_c[60]].map(dc_tr3)

#mapping Electrical
er1=list(h.groupby(d_c[42]).SalePrice.mean().sort_values().index)
dc_er1={er1[0]:1,er1[1]:2,er1[2]:3,er1[3]:4,er1[4]:5}
test_df[d_c[42]]=test_df[d_c[42]].map(dc_er1)
test_df[d_c[42]]=test_df[d_c[42]].fillna(0)



plt.figure(figsize = (9, 5)) 
df['SalePrice'].plot(kind ="hist") 