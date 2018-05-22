
# coding: utf-8

# # scikit-learn入门

# * 准备数据集

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


X = np.random.randint(0, 100, (10, 4))
y = np.random.randint(0, 3, 10)
y.sort()

print('样本：')
print(X)
print('标签：', y)


# In[3]:


# 分割训练集、测试集
# random_state确保每次随机分割得到相同的结果
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3., random_state=7) 

print('训练集：')
print(X_train)
print(y_train)

print('测试集：')
print(X_test)
print(y_test)


# In[4]:


# 特征归一化
from sklearn import preprocessing

x1 = np.random.randint(0, 1000, 5).reshape(5,1)
x2 = np.random.randint(0, 10, 5).reshape(5, 1)
x3 = np.random.randint(0, 100000, 5).reshape(5, 1)

X = np.concatenate([x1, x2, x3], axis=1)
print(X)


# In[5]:


print(preprocessing.scale(X))


# In[9]:


# 生成分类数据进行验证scale的必要性
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2, 
                           random_state=25, n_clusters_per_class=1, scale=100)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()


# In[10]:


from sklearn import svm

# 注释掉以下这句表示不进行特征归一化     
# X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3., random_state=7) 
svm_classifier = svm.SVC()
svm_classifier.fit(X_train, y_train)
svm_classifier.score(X_test, y_test)


# * 训练模型

# In[11]:


# 回归模型
from sklearn import datasets

boston_data = datasets.load_boston()
X = boston_data.data
y = boston_data.target

print('样本：')
print(X[:5, :])
print('标签：')
print(y[:5])


# In[12]:


# 选择线性回顾模型
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()


# In[13]:


from sklearn.model_selection import train_test_split

# 分割训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3., random_state=7) 


# In[14]:


# 训练模型
lr_model.fit(X_train, y_train)


# In[15]:


# 返回参数
lr_model.get_params()


# In[16]:


lr_model.score(X_train, y_train)


# In[17]:


lr_model.score(X_test, y_test)


# * 交叉验证

# In[18]:


from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3., random_state=10) 

k_range = range(1, 31)
cv_scores = []
for n in k_range:
    knn = KNeighborsClassifier(n)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy') # 分类问题使用
    #scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='neg_mean_squared_error') # 回归问题使用
    cv_scores.append(scores.mean())
    
plt.plot(k_range, cv_scores)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()


# In[31]:


# 选择最优的K
best_knn = KNeighborsClassifier(n_neighbors=5)
best_knn.fit(X_train, y_train)
print(best_knn.score(X_test, y_test))
print(best_knn.predict(X_test))

