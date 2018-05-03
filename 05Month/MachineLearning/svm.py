from sklearn import svm

x=[[2,0],[1,1],[2,3]]
y=[0,0,1]
clf= svm.SVC(kernel='linear')
clf.fit(x,y)

print(clf)
print(clf.support_vectors_) #support vector
print(clf.support_) #support index
print(clf.n_support_) #support class label numnber

predictLabel = clf.predict([-1,2])
print(predictLabel)
