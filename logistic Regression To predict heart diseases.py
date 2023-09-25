#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[2]:


heart_data=pd.read_csv("framingham.csv")
heart_data.head(15)


# In[3]:


sns.countplot(x="prevalentStroke",data=heart_data)


# In[4]:


sns.countplot(x="prevalentHyp",data=heart_data)


# In[5]:


heart_data["age"].plot.hist()


# In[6]:


heart_data["cigsPerDay"].plot.hist()


# In[7]:


heart_data["glucose"].plot.hist()


# In[8]:


sns.countplot(x="TenYearCHD",data=heart_data)


# In[9]:


heart_data.isnull()


# In[10]:


heart_data.isnull().sum()


# In[11]:


sns.heatmap(heart_data.isnull(),yticklabels=False,cmap="viridis")


# In[12]:


heart_data.drop("education",axis=1,inplace=True)
heart_data.head()


# In[13]:


m=np.mean(heart_data["glucose"])
print(m)


# In[15]:


heart_data["glucose"].replace(to_replace=np.nan,value=m,inplace=True)
heart_data["BMI"].replace(to_replace=np.nan,value=25,inplace=True)
heart_data["cigsPerDay"].replace(to_replace=np.nan,value=5,inplace=True)
heart_data["heartRate"].replace(to_replace=np.nan,value=90,inplace=True)
heart_data.drop("currentSmoker",axis=1,inplace=True)


# In[16]:


heart_data["glucose"].plot.hist()


# In[17]:


heart_data.head(10)


# In[18]:


sns.heatmap(heart_data.isnull(),yticklabels=False,cmap="viridis")


# In[19]:


heart_data["totChol"].replace(to_replace=np.nan,value=247.0,inplace=True)
heart_data.drop("BPMeds",axis=1,inplace=True)


# In[20]:


sns.heatmap(heart_data.isnull(),yticklabels=False,cmap="viridis")


# In[21]:


heart_data.isnull().sum()


# In[22]:


sns.countplot(x="TenYearCHD",data=heart_data)


# In[23]:


X=heart_data.drop("TenYearCHD",axis=1)
y=heart_data["TenYearCHD"].values
y


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=15)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.datasets import load_iris
from sklearn import preprocessing
iris = load_iris()
print(iris.data.shape)
X = iris.data
y = iris.target
normalized_X = preprocessing.normalize (X)


# In[34]:


print('Accuracy Score :' + str(accuracy_score(y_test,y_pred)))
print('Precision Score :' + str(precision_score(y_test,y_pred)))
print('Recall Score :' + str(recall_score(y_test,y_pred)))
print('F1 Score :' + str(f1_score(y_test,y_pred)))
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))


# In[37]:


from sklearn.model_selection import GridSearchCV
clf= LogisticRegression()
grid_values = {'penalty': ['l1','l2'],'C': [0.001,.009,0.01,.09,1,5,10,25,15,100]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'accuracy')
grid_clf_acc.fit(X_train, y_train)
y_pred = grid_clf_acc.predict(X_test)


# In[38]:


accuracy=accuracy_score(y_test,y_pred)
accuracy


# In[ ]:




