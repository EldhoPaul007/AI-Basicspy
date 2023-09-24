#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv("Salary.csv")


# In[4]:


data.head(10)


# In[6]:


plt.scatter(data['YearsExperience'],data['Salary'])
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show


# x=data.iloc[:,:-1].values
# y=data.iloc[:,-1].values

# In[12]:


y


# In[16]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state = 0)


# In[19]:


x_test.size


# In[20]:


x_train


# In[21]:


y_train


# In[22]:


y_test


# In[23]:


from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(x_train,y_train)


# In[24]:


y_pred_LR=LR.predict(x_test)


# In[25]:


y_pred_LR


# In[32]:


diff_LR=y_pred_LR


# In[33]:


res_df=pd.concat([pd.Series(y_pred_LR),pd.Series(y_test),pd.Series(diff_LR)],axis=1)
res_df.columns=['Prediction', 'Original Data', 'Diff']


# In[34]:


res_df


# In[38]:


plt. scatter(x_train,y_train,color='blue')
plt.plot(x_train,LR.predict(x_train),color='red')
plt.title("Salary VS Experience(Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# In[40]:


plt. scatter(x_test,y_test,color='blue')
plt.plot(x_train,LR.predict(x_train),color='red')
plt.title("Salary VS Experience(Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# In[43]:


from sklearn import metrics
rmse= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
R2=metrics.r2_score(y_test,y_pred_LR)


# In[44]:


rmse


# In[45]:


R2


# In[82]:


LR.predict([[2]])


# In[ ]:




