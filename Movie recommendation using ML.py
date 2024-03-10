#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[18]:


movies_data = pd.read_csv('movies.csv')


# In[19]:


movies_data.head()


# In[20]:


movies_data.shape


# In[21]:


selected_features =['genres','keywords','tagline','cast','director']
print(selected_features)


# In[22]:


for features in selected_features:
    movies_data[features]=movies_data[features].fillna('')


# In[26]:


combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[27]:


print(combined_features)


# In[28]:


vectorizer = TfidfVectorizer()


# In[29]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[31]:


print(feature_vectors)


# In[32]:


similarity = cosine_similarity(feature_vectors)


# In[33]:


print(similarity)


# In[34]:


print(similarity.shape)


# In[35]:


movie_name = input('Enter your favorate movie name :')


# In[36]:


list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[39]:


find_close_match = difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)


# In[41]:


close_match = find_close_match[0]
print(close_match)


# In[44]:


index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[45]:


similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[46]:


len(similarity_score)


# In[47]:


sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[52]:


print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<21):
    print(i, '.',title_from_index)
    i+=1


# In[56]:


#run it here
movie_name = input('Enter your favorate movie name :')
list_of_all_titles = movies_data['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name,list_of_all_titles)
close_match = find_close_match[0]
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<21):
    print(i, '.',title_from_index)
    i+=1


# In[ ]:




