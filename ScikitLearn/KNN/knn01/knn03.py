#为了完成找到两组数据集中最近邻点的简单任务, 可以使用 sklearn.neighbors 中的无监督算法:
from sklearn.neighbors import NearestNeighbors
import numpy as np
#生成数组
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

distances, indices = nbrs.kneighbors(X)

print(indices)
'''
输出：
[[0 1]
 [1 0]
 [2 1]
 [3 4]
 [4 3]
 [5 4]]
'''
print(distances)
'''
输出：
[[0.         1.        ]
 [0.         1.        ]
 [0.         1.41421356]
 [0.         1.        ]
 [0.         1.        ]
 [0.         1.41421356]]
'''
print(nbrs.kneighbors_graph(X).toarray())
'''
输出：
[[1. 1. 0. 0. 0. 0.]
 [1. 1. 0. 0. 0. 0.]
 [0. 1. 1. 0. 0. 0.]
 [0. 0. 0. 1. 1. 0.]
 [0. 0. 0. 1. 1. 0.]
 [0. 0. 0. 0. 1. 1.]]
'''