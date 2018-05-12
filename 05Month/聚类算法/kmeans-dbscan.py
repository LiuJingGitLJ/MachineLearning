
# coding: utf-8

# # 聚类

# In[2]:


# beer dataset
import pandas as pd
beer = pd.read_csv('data.txt', sep=' ')
beer


# In[3]:


X = beer[["calories","sodium","alcohol","cost"]]


# ## K-means clustering

# In[4]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=3).fit(X)
km2 = KMeans(n_clusters=2).fit(X)


# In[5]:


km.labels_


# In[10]:


beer['cluster'] = km.labels_
beer['cluster2'] = km2.labels_
beer.sort_values('cluster')


# In[11]:


from pandas.tools.plotting import scatter_matrix
# get_ipython().magic('matplotlib inline')

cluster_centers = km.cluster_centers_

cluster_centers_2 = km2.cluster_centers_


# In[12]:


beer.groupby("cluster").mean()


# In[13]:


beer.groupby("cluster2").mean()


# In[14]:


centers = beer.groupby("cluster").mean().reset_index()


# In[15]:


# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14


# In[16]:


import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])


# In[17]:


plt.scatter(beer["calories"], beer["alcohol"],c=colors[beer["cluster"]])

plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')

plt.xlabel("Calories")
plt.ylabel("Alcohol")


# In[18]:


scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster"]], figsize=(10,10))
plt.suptitle("With 3 centroids initialized")


# In[19]:


scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster2"]], figsize=(10,10))
plt.suptitle("With 2 centroids initialized")


# ### Scaled data

# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[21]:


km = KMeans(n_clusters=3).fit(X_scaled)


# In[22]:


beer["scaled_cluster"] = km.labels_
beer.sort_values("scaled_cluster")


# What are the "characteristics" of each cluster?

# In[23]:


beer.groupby("scaled_cluster").mean()


# In[24]:


pd.scatter_matrix(X, c=colors[beer.scaled_cluster], alpha=1, figsize=(10,10), s=100)


# ## 聚类评估：轮廓系数（Silhouette Coefficient ）
# 
# <img src="1.png" alt="FAO" width="490">
# 
# - 计算样本i到同簇其他样本的平均距离ai。ai 越小，说明样本i越应该被聚类到该簇。将ai 称为样本i的簇内不相似度。
# - 计算样本i到其他某簇Cj 的所有样本的平均距离bij，称为样本i与簇Cj 的不相似度。定义为样本i的簇间不相似度：bi =min{bi1, bi2, ..., bik}
# 
# 
# * si接近1，则说明样本i聚类合理
# * si接近-1，则说明样本i更应该分类到另外的簇
# * 若si 近似为0，则说明样本i在两个簇的边界上。

# In[25]:


from sklearn import metrics
score_scaled = metrics.silhouette_score(X,beer.scaled_cluster)
score = metrics.silhouette_score(X,beer.cluster)
print(score_scaled, score)


# In[26]:


scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score)

scores


# In[27]:


plt.plot(list(range(2,20)), scores)
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")


# ##  DBSCAN clustering

# In[28]:


from sklearn.cluster import DBSCAN
db = DBSCAN(eps=10, min_samples=2).fit(X)


# In[29]:


labels = db.labels_


# In[32]:


beer['cluster_db'] = labels
beer.sort_values('cluster_db')


# In[33]:


beer.groupby('cluster_db').mean()


# In[34]:


pd.scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10,10), s=100)

