#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ZhengzhengLiu

#聚类算法评估

import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

#解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#初始化三个中心
centers = [[1,1],[-1,-1],[1,-1]]
clusters = len(centers)     #聚类数目为3
#产生3000组二维数据样本，三个中心点，标准差是0.7
X,Y = make_blobs(n_samples=300,centers=centers,cluster_std=0.7,random_state=28)

#构建kmeans算法
k_means =  KMeans(init="k-means++",n_clusters=clusters,random_state=28)
t0 = time.time()
k_means.fit(X)      #模型训练
km_batch = time.time()-t0       #使用kmeans训练数据消耗的时间
print("K-Means算法模型训练消耗时间:%.4fs"%km_batch)

#构建mini batch kmeans算法
batch_size = 100        #采样集的大小
mbk = MiniBatchKMeans(init="k-means++",n_clusters=clusters,batch_size=batch_size,random_state=28)
t0 = time.time()
mbk.fit(X)
mbk_batch = time.time()-t0
print("Mini Batch K-Means算法模型训练消耗时间:%.4fs"%mbk_batch)

km_y_hat = k_means.labels_
mbkm_y_hat = mbk.labels_

k_means_cluster_centers = k_means.cluster_centers_
mbk_means_cluster_centers = mbk.cluster_centers_
print ("K-Means算法聚类中心点:\ncenter=", k_means_cluster_centers)
print ("Mini Batch K-Means算法聚类中心点:\ncenter=", mbk_means_cluster_centers)
order = pairwise_distances_argmin(k_means_cluster_centers,
                                  mbk_means_cluster_centers)

#效果评估
### 效果评估
score_funcs = [
    metrics.adjusted_rand_score,    #ARI（调整兰德指数）
    metrics.v_measure_score,        #均一性与完整性的加权平均
    metrics.adjusted_mutual_info_score, #AMI（调整互信息）
    metrics.mutual_info_score,      #互信息
]

## 2. 迭代对每个评估函数进行评估操作
for score_func in score_funcs:
    t0 = time.time()
    km_scores = score_func(Y, km_y_hat)
    print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (score_func.__name__, km_scores, time.time() - t0))

    t0 = time.time()
    mbkm_scores = score_func(Y, mbkm_y_hat)
    print("Mini Batch K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs\n" % (score_func.__name__, mbkm_scores, time.time() - t0))

#运行结果：
# K-Means算法模型训练消耗时间:0.6350s
# Mini Batch K-Means算法模型训练消耗时间:0.0900s
# K-Means算法聚类中心点:
# center= [[ 0.96091862  1.13741775]
#  [ 1.1979318  -1.02783007]
#  [-0.98673669 -1.09398768]]
# Mini Batch K-Means算法聚类中心点:
# center= [[ 1.34304199 -1.01641075]
#  [ 0.83760683  1.01229021]
#  [-0.92702179 -1.08205992]]
# K-Means算法:adjusted_rand_score评估函数计算结果值:0.72566；计算消耗时间:0.071s
# Mini Batch K-Means算法:adjusted_rand_score评估函数计算结果值:0.69544；计算消耗时间:0.001s
#
# K-Means算法:v_measure_score评估函数计算结果值:0.67529；计算消耗时间:0.004s
# Mini Batch K-Means算法:v_measure_score评估函数计算结果值:0.65055；计算消耗时间:0.004s
#
# K-Means算法:adjusted_mutual_info_score评估函数计算结果值:0.67263；计算消耗时间:0.006s
# Mini Batch K-Means算法:adjusted_mutual_info_score评估函数计算结果值:0.64731；计算消耗时间:0.005s
#
# K-Means算法:mutual_info_score评估函数计算结果值:0.74116；计算消耗时间:0.002s
# Mini Batch K-Means算法:mutual_info_score评估函数计算结果值:0.71351；计算消耗时间:0.001s
