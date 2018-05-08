from sklearn.neighbors import KNeighborsClassifier

X = [[0], [1], [2], [3],[4], [5],[6],[7],[8]]
y = [0, 0, 0, 1, 1, 1, 2, 2, 2]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(neigh.predict([[1.1]]))
print(neigh.predict([[1.6]]))
print(neigh.predict([[5.2]]))
print(neigh.predict([[5.8]]))
print(neigh.predict([[6.2]]))