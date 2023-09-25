#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot  as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


dataset = pd.read_csv('dataUSA.csv')
dataset.info()


# In[59]:


dataset.shape


# In[60]:


dataset.head()


# In[61]:


dataset.drop(['date'], axis = 1, inplace = True)
dataset.head()


# **Checking how many different Countries are there**

# In[62]:


dataset.country.value_counts()


# In[63]:


dataset.drop(['country'], axis = 1, inplace = True)
dataset.head()


# Since we already have statezip, we can safely delete street and city.

# In[64]:


dataset.drop(['street', 'city'], axis = 1, inplace = True)
dataset.head()


# In[65]:


dataset.isnull().sum()


# In[66]:


a4_dims = (10, 8)
fig, ax = plt.subplots(figsize=a4_dims)
cor = dataset.corr()
sns.heatmap(cor, annot = True, cmap="YlGnBu")


# In[67]:


a4_dims = (15, 5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(x = dataset.bedrooms, y = dataset.price)


# In[68]:


dataset.groupby('bedrooms').price.agg([len, min, max])


# In[69]:


df = dataset[(dataset.bedrooms > 0) & (dataset.bedrooms < 9)].copy()


# In[70]:


df.shape


# In[71]:


df.statezip.value_counts()


# In[72]:


a4_dims = (5, 18)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(ax = ax, x = df.price, y = df.statezip)


# In[73]:


a4_dims = (15, 8)
fig, ax = plt.subplots(figsize=a4_dims)
sns.distplot(a = df.price, bins = 1000, color = 'r', ax = ax)


# In[74]:


df.price.agg([min, max])


# In[75]:


len(df[(df.price == 0)])


# In[76]:


a4_dims = (15, 5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(x = df.bathrooms, y = df.price)


# In[77]:


zero_price = df[(df.price == 0)].copy()
zero_price.shape


# In[78]:


zero_price.head()


# In[79]:


sns.distplot(zero_price.sqft_living)


# In[80]:


zero_price.agg([min, max, 'mean', 'median'])


# In[81]:


sim_from_ori = df[(df.bedrooms == 4) & (df.bathrooms > 1) & (df.bathrooms < 4) & (df.sqft_living > 2500) & (df.sqft_living < 3000) & (df.floors < 3) & (df.yr_built < 1970)].copy()


# In[82]:


sim_from_ori.shape


# In[83]:


sim_from_ori.head()


# In[84]:


sim_from_ori.price.mean()


# In[85]:


yr_sqft = df[(df.sqft_living > 2499) & (df.sqft_living < 2900)].copy()
yr_price_avg = yr_sqft.groupby('yr_built').price.agg('mean')


# In[86]:


plt.plot(yr_price_avg)


# In[87]:


df.price.replace(to_replace = 0, value = 735000, inplace = True)
len(df[(df.price == 0)])


# In[88]:


df.head()


# In[89]:


df.drop(['sqft_above'], axis = 1, inplace = True)
df.shape


# In[90]:


df = df.reset_index()
df.info()


# In[91]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[92]:


df['statezip_encoded'] = le.fit_transform(df.statezip)
df.head()


# In[93]:


df.statezip_encoded.value_counts()


# In[94]:


df.drop(['statezip'], axis = 1, inplace = True)
df.head()


# In[95]:


from sklearn.preprocessing import OneHotEncoder
ohc = OneHotEncoder()


# In[96]:


ohc_df = pd.DataFrame(ohc.fit_transform(df[['statezip_encoded']]).toarray())
# ohc_df = ohc_df.astype(int)
ohc_df.head()


# In[97]:


df = df.join(ohc_df)
df.head()


# In[98]:


df.tail()


# In[99]:


df.drop(['statezip_encoded'], axis = 1, inplace = True)


# In[100]:


df.info


# In[101]:


df.shape


# In[102]:


X = df.iloc[:, 1:]
X.shape


# In[103]:


y = df.price


# In[128]:


from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=.1, random_state=0)


# In[129]:


print(len(X_train) / len(df))


# In[130]:


X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)
print(len(X_test) / len(y_rem))


# In[131]:


print(len(X_train))
print(len(X_val))
print(len(X_val))


# In[132]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()


# In[125]:


lin_reg.fit(X_train,y_train)


# In[110]:


from sklearn.metrics import mean_squared_error
y_pred = lin_reg.predict(X_val)
mse = mean_squared_error(y_pred, y_val)
rmse = np.sqrt(mse)
rmse


# In[111]:


y_val.head(10)


# In[112]:


y_pred


# In[113]:


y_pred_test = lin_reg.predict(X_test)
mse = mean_squared_error(y_pred_test, y_test)
rmse = np.sqrt(mse)
rmse


# In[114]:


lin_reg.score(X_test, y_test)


# In[115]:


y_test


# In[116]:


y_pred_test


# In[117]:


from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(random_state = 42, max_depth = 10)


# In[118]:


reg.fit(X_train, y_train)


# In[119]:


reg.score(X_test, y_test)


# In[120]:


y_val.head(10)


# In[121]:


y_pred_dt


# In[ ]:




