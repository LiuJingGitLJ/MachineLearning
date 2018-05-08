from sklearn import datasets
#导入内置数据集模块
from sklearn.neighbors import KNeighborsClassifier
#导入sklearn.neighbors模块中KNN类
import numpy as np
iris=datasets.load_iris()
# print(iris)
#导入鸢尾花的数据集，iris是一个数据集，内部有样本数据
iris_x=iris.data
iris_y=iris.target

indices = np.random.permutation(len(iris_x))
#permutation接收一个数作为参数(150),产生一个0-149一维数组，只不过是随机打乱的
iris_x_train = iris_x[indices[:-10]]
 #随机选取140个样本作为训练数据集
iris_y_train = iris_y[indices[:-10]]
# 并且选取这140个样本的标签作为训练数据集的标签
iris_x_test = iris_x[indices[-10:]]
# 剩下的10个样本作为测试数据集
iris_y_test = iris_y[indices[-10:]]
# 并且把剩下10个样本对应标签作为测试数据及的标签

knn = KNeighborsClassifier()
# 定义一个knn分类器对象
knn.fit(iris_x_train, iris_y_train)
# 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
iris_y_predict = knn.predict(iris_x_test)
# 调用该对象的测试方法，主要接收一个参数：测试数据集
score = knn.score(iris_x_test, iris_y_test, sample_weight=None)
# 调用该对象的打分方法，计算出准确率


print('iris_y_predict = ')
print(iris_y_predict)
# 输出测试的结果
print('iris_y_test = ')
print(iris_y_test)
# 输出原始测试数据集的正确标签，以方便对比
print('Accuracy:', score)
# 输出准确率计算结果</span>
'''
iris_y_predict = 
[2 0 2 2 2 0 0 1 2 0]
iris_y_test = 
[2 0 2 2 2 0 0 1 1 0]
Accuracy: 0.9
'''
