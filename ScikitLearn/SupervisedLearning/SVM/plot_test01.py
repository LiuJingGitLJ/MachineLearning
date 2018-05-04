

#coding:utf8
#导入svm的库
from sklearn import svm
x = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]  #对应x的分类标记
clf = svm.SVC(kernel= 'linear') #线性核函数
clf.fit(x, y)

print (clf)
print (clf.support_vectors_ ) #支持向量
print (clf.support_ ) #支持向量是哪几个(下标)
print (clf.n_support_)    #每一类中有几个支持向量
'''
print (clf.predict([0, 1]))   #测试数据
ValueError: Expected 2D array, got 1D array instead:
array=[0. 1.].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
'''
print (clf.predict([[0, 1]]))   #测试数据

print("########################################\n")
import numpy as np
import pylab as pl

#生成随机点数据集
np.random.seed(0) #固定随机值
x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] *20 +[1] * 20

print(x)
print(y)

clf2 = svm.SVC(kernel='linear')
clf2.fit(x, y)
print(clf2.support_)
#画出散点图
#画出支持向量的点，参数：x，y，大小
pl.scatter(clf2.support_vectors_[:,0],clf2.support_vectors_[:,1],s=80)
#画出全部的点，参数：x，y，颜色，colormap，形状
pl.scatter(x[:,0],x[:,1],c=y,cmap=pl.cm.Paired,marker='o')
pl.axis('tight')
#pl.savefig("dd") 保存绘图
pl.show()