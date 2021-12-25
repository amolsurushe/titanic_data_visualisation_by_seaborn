#!/usr/bin/env python
# coding: utf-8

# In[219]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[220]:


sns.set_style('whitegrid')


# In[221]:


titanic = sns.load_dataset('titanic')


# In[230]:


titanic.head()


# In[223]:


titanic.info()


# In[224]:


for i in titanic.columns:
    null_rate = titanic[i].isna().sum()/len(titanic) * 100
    if null_rate > 0 :
        print("{} null rate: {}%".format(i,round(null_rate,2)))


# In[ ]:





# In[225]:


titanic['embarked'].replace(np.nan,'No data',inplace=True)


# In[226]:


titanic['embark_town'].replace(np.nan,'No data',inplace=True)


# In[227]:


titanic


# In[229]:


sns.jointplot(x='fare',y='age',data=titanic)


# In[231]:


sns.displot(titanic['fare'],bins=30,kde=False,color='red')


# In[232]:


sns.boxplot(x='class',y='age',data=titanic,palette='rainbow')


# In[233]:


sns.stripplot(x='class',y='age',data=titanic,palette='Set2')


# In[234]:


sns.countplot(x='sex',data=titanic)


# In[235]:


titanic.head()


# In[236]:


sns.heatmap(titanic.corr(),cmap='coolwarm',annot=True)
plt.title('titanic.corr()')


# In[237]:


titanic = sns.load_dataset('titanic')
titanic.groupby(['survived']).count()


# In[238]:


g = sns.FacetGrid(data=titanic,col='sex')
g.map(plt.hist,'age')

