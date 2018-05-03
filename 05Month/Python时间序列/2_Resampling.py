
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ###  数据重采样 ###
# - 时间数据由一个频率转换到另一个频率
# - 降采样
# - 升采样

# In[7]:


rng = pd.date_range('1/1/2011', periods=90, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts.head()


# In[8]:


ts.resample('M').sum()


# In[9]:


ts.resample('3D').sum()


# In[13]:


day3Ts = ts.resample('3D').mean()
day3Ts


# In[14]:


print(day3Ts.resample('D').asfreq())


# ###  插值方法： ###
# - ffill 空值取前面的值
# - bfill 空值取后面的值
# - interpolate 线性取值

# In[26]:


day3Ts.resample('D').ffill(1)


# In[23]:


day3Ts.resample('D').bfill(1)


# In[24]:


day3Ts.resample('D').interpolate('linear')

