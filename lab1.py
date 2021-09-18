import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import statistics
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from imblearn.under_sampling import NearMiss
from collections import Counter

class Python:
  def __init__(self,dt2,dt3,dt4,dt5):
    self.df=pd.read_csv(dt2)
    self.dt3=pd.read_csv(dt3)
    self.dt4=pd.read_csv(dt4)
    self.dt5=pd.read_csv(dt5)
  def titanic(self):
    dt1=self.df
    label_encoder=preprocessing.LabelEncoder()
    dt1['Sex']=label_encoder.fit_transform(dt1['Sex'])
    dt1['Survived']=dt1['Survived'].astype(str)
    dt1['survived']=dt1['Survived'].apply(lambda x: '-1' if x == '0' else '+1' )
    c=(pd.get_dummies(dt1.survived))
    c['-1'].astype(int)
    c['+1'].astype(int)
    dt1=pd.concat([dt1,c],axis=1)   
    d=dt1[['Age','Fare']]
    x=preprocessing.normalize(d)
    scaled_df = pd.DataFrame(x,columns=d.columns)
    dt1=dt1.drop(['Age','Fare'],axis=1)
    dt1=pd.concat([dt1,scaled_df],axis=1)
    x=dt1.drop(['survived','Name','Survived','+1','-1'],axis='columns')
    y=dt1['survived']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    model=DecisionTreeClassifier(criterion='entropy',max_depth=1)
    adaboost=AdaBoostClassifier(base_estimator=model,n_estimators=400,learning_rate=1)
    model=adaboost.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    self.dt=dt1
    print('accuracy:',metrics.accuracy_score(y_test,y_pred))
  def titan_output(self):
    dt1=self.dt
    died=dt1[dt1['-1']== 1]
    not_died=dt1[dt1['+1']==1]
    print('length of Death',len(died))
    pd.set_option('display.max_columns',None)
    print(died.head())
    print('length of survived',len(not_died))
    print(not_died.head())

  def albanone(self):
    dt2=self.dt3
    label_encoder=preprocessing.LabelEncoder()
    dt2['Class']=label_encoder.fit_transform(dt2['Class'])
    dt2['Sex']=label_encoder.fit_transform(dt2['Sex'])
    dt2.isnull().sum()
    k=statistics.median(dt2.Class.unique())
    sns.countplot(x=dt2['Class'],data=dt2,palette='hls')
    plt.show()
    plt.hist(dt2['Class'])
    dt2['new_class']=dt2['Class'].apply(lambda x:'-1' if x<k else '+1' )
    c0,c1=dt2['new_class'].value_counts()
    cd=dt2[dt2['new_class']=='+1']
    cp=dt2[dt2['new_class']=='-1']
    cd_over=cd.sample(c0,replace=True)
    test1=pd.concat([cd_over,cp],axis=0)
    test1['new_class'].value_counts()
    x=test1.drop(['new_class','Class'],axis=1)
    y=test1['new_class']
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4)
    model=LogisticRegression()
    mod=model.fit(x_train,y_train)
    y_pred=mod.predict(x_test)
    print(metrics.accuracy_score(y_test,y_pred))
    print('length of +1',len(cd))
    print('class +1',cd.head())
    print('length of -1',len(cp))
    print('class -1',cp.head())

  def german(self):
    ger=self.dt4
    d=[]
    label_encoder=preprocessing.LabelEncoder()
    for i in ger.columns:
      if ger[i].dtype == object:
        ger[i]=label_encoder.fit_transform(ger[i])
    corr=ger.corr()
    sns.heatmap(corr)
    c=np.full((corr.shape[0],),True,dtype=bool)
    for i in range(corr.shape[0]):
      for j in range(i+1,corr.shape[0]):
        if corr.iloc[i,j] >= 0.5:
          if c[j]:
            c[j]=False
    s=ger.columns[c]
    ger=ger[s]
    ger.shape
    x=ger.drop(['Class'],axis=1)
    y=ger['Class']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
    rf=RandomForestClassifier(random_state=42,max_depth=5,n_estimators=400,n_jobs=-1,oob_score=True)
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_test)
    print(metrics.accuracy_score(y_test,y_pred))
    print('german class 1 len',len(ger[ger['Class']==1]))
    print(ger[ger['Class']==1].head())
    print('german class 2 len',len(ger[ger['Class']==2]))
    print(ger[ger['Class']==2].head())
    
  def cancer(self):
    #splitting based on index.
    bc=self.dt5
    idm=bc[bc.index % 5 ==0]
    idn=bc[bc.index % 5 !=0]
    xtrain=idm.drop(['Class'],axis=1)
    xtest=idn.drop(['Class'],axis=1)
    ytrain=idm['Class']
    ytest=idn['Class']
    r=idm[idm['Class']=='no-recurrence-events']
    n=idn[idn['Class']=='recurrence-events']
    print('length of no recurrence',len(r))
    print('No-recurrence events',r.head())
    print('length of  recurrence',len(n))
    print('recurrence events',n.head())
    

if __name__=='__main__':
  model_instance=Python('/home/anusha/first_class/titanic.csv','/home/anusha/first_class/abalone.csv','/home/anusha/first_class/german.csv','/home/anusha/first_class/breast_cancer.csv')     
print('choose the file number:')
print('1:titanic')
print('2:albanone')
print('3:german')
print('4:breast_cancer')
c=str(input('Énter the number '))
if c == '1':
  model_instance.titanic()
  model_instance.titan_output()
if c == '2':
  model_instance.albanone()
if c == '3':
  model_instance.german() 
if c == '4':
  model_instance.cancer() 
