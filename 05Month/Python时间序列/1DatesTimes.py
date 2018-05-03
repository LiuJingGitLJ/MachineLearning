
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ###  时间序列 ###
# - 时间戳（timestamp）
# - 固定周期（period）
# - 时间间隔（interval）
# 
# <img src="f1.png" alt="FAO" width="590" >

# ### date_range ###
# - 可以指定开始时间与周期
# - H：小时
# - D：天
# - M：月

# In[58]:


# TIMES #2016 Jul 1 7/1/2016 1/7/2016 2016-07-01 2016/07/01
rng = pd.date_range('2016-07-01', periods = 10, freq = '3D')
rng


# In[26]:


time=pd.Series(np.random.randn(20),
           index=pd.date_range(dt.datetime(2016,1,1),periods=20))
print(time)


# ###  truncate过滤 ###

# In[50]:


time.truncate(before='2016-1-10')


# In[51]:


time.truncate(after='2016-1-10')


# In[27]:


print(time['2016-01-15'])


# In[28]:


print(time['2016-01-15':'2016-01-20'])


# In[29]:


data=pd.date_range('2010-01-01','2011-01-01',freq='M')
print(data)


# <img src="f2.png" alt="FAO" width="590" >

# In[17]:


#时间戳
pd.Timestamp('2016-07-10')


# In[18]:


# 可以指定更多细节
pd.Timestamp('2016-07-10 10')


# In[19]:


pd.Timestamp('2016-07-10 10:15')


# In[ ]:


# How much detail can you add?


# In[ ]:


t = pd.Timestamp('2016-07-10 10:15')


# In[30]:


# 时间区间
pd.Period('2016-01')


# In[21]:


pd.Period('2016-01-01')


# In[35]:


# TIME OFFSETS
pd.Timedelta('1 day')


# In[36]:


pd.Period('2016-01-01 10:10') + pd.Timedelta('1 day')


# In[37]:


pd.Timestamp('2016-01-01 10:10') + pd.Timedelta('1 day')


# In[38]:


pd.Timestamp('2016-01-01 10:10') + pd.Timedelta('15 ns')


# In[39]:


p1 = pd.period_range('2016-01-01 10:10', freq = '25H', periods = 10)


# In[40]:


p2 = pd.period_range('2016-01-01 10:10', freq = '1D1H', periods = 10)


# In[41]:


p1


# In[42]:


p2


# In[43]:


# 指定索引
rng = pd.date_range('2016 Jul 1', periods = 10, freq = 'D')
rng
pd.Series(range(len(rng)), index = rng)


# In[44]:


periods = [pd.Period('2016-01'), pd.Period('2016-02'), pd.Period('2016-03')]
ts = pd.Series(np.random.randn(len(periods)), index = periods)
ts


# In[45]:


type(ts.index)


# In[46]:


# 时间戳和时间周期可以转换
ts = pd.Series(range(10), pd.date_range('07-10-16 8:00', periods = 10, freq = 'H'))
ts


# In[47]:


ts_period = ts.to_period()
ts_period


# In[48]:


ts_period['2016-07-10 08:30':'2016-07-10 11:45'] 


# In[49]:


ts['2016-07-10 08:30':'2016-07-10 11:45'] 

