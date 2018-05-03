
# coding: utf-8

# In[2]:


get_ipython().magic('matplotlib inline')
import matplotlib.pylab
import numpy as np
import pandas as pd


# In[31]:


df = pd.Series(np.random.randn(600), index = pd.date_range('7/1/2016', freq = 'D', periods = 600))


# In[32]:


df.head()


# In[33]:


r = df.rolling(window = 10)
r


# In[34]:


#r.max, r.median, r.std, r.skew, r.sum, r.var
print(r.mean())


# In[35]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure(figsize=(15, 5))

df.plot(style='r--')
df.rolling(window=10).mean().plot(style='b')

