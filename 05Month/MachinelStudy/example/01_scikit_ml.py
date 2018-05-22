
# coding: utf-8

# # 通过scikit-learn认识机器学习

# * 加载示例数据集

# In[1]:


from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()


# In[2]:


# 查看数据集
# iris
print(iris.data)
print(iris.data.shape)
print(iris.target_names)
print(iris.target)


# In[3]:


# digits
print(digits.data)
print(digits.data.shape)
print(digits.target_names)
print(digits.target)


# * 在训练集上训练模型

# In[14]:


# 手动划分训练集、测试集 
n_test = 100 # 测试样本个数
train_X = digits.data[:-n_test, :]
train_y = digits.target[:-n_test]

test_X = digits.data[-n_test:, :]
y_true = digits.target[-n_test:]


# In[15]:


# 选择SVM模型
from sklearn import svm

svm_model = svm.SVC(gamma=0.001, C=100.)
# svm_model = svm.SVC(gamma=100., C=1.)

# 训练模型
svm_model.fit(train_X, train_y)


# In[16]:


# 选择LR模型
from sklearn.linear_model import LogisticRegression
# 初始化模型
lr_model = LogisticRegression()
# 训练模型
lr_model.fit(train_X, train_y)


# * 在测试集上测试模型

# In[17]:


y_pred_svm = svm_model.predict(test_X)
y_pred_lr = lr_model.predict(test_X)


# In[19]:


# 查看结果
from sklearn.metrics import accuracy_score

#print '预测标签：', y_pred
#print '真实标签：', y_true

print('SVM结果：', accuracy_score(y_true, y_pred_svm))
print('LR结果：', accuracy_score(y_true, y_pred_lr))


# * 保存模型

# In[20]:


import pickle

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)


# In[21]:


import numpy as np

# 重新加载模型进行预测
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

random_samples_index = np.random.randint(0, 1796, 5)
random_samples = digits.data[random_samples_index, :]
random_targets = digits.target[random_samples_index]

random_predict = model.predict(random_samples)

print(random_predict)
print(random_targets)

