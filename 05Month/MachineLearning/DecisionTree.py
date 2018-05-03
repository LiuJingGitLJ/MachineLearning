# -×- encoding=utf-8 -*-
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

#read the csv file
allElectronicsData = open('play.csv','rb')
reader = csv.reader(allElectronicsData)
header=reader.next()

print(header)

featureList=[]
labelList=[]

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict={}
    for i in range(1,len(row)-1):
        rowDict[header[i]]=row[i]
        
    featureList.append(rowDict)

print(featureList)

#vectorize feature
vec = DictVectorizer()
dumpyX = vec.fit_transform(featureList).toarray()

print("dunmpyX "+ str(dumpyX) )
print("feature_name"+str(vec.get_feature_names()))

print("labelList "+ str(labelList))

#vectorize class lables
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:" +str(dummyY))


#use the decision tree for classfication

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dumpyX,dummyY) #构造决策树

#打印构造决策树采用的参数
print("clf : "+str(clf))

#visilize the model

#with open('play.dot','w') as f:
#    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names,out_file=f)
# dot -Tpdf in.dot -o out.pdf输出pdf文件

#验证数据，取一行数据，修改几个属性预测结果
oneRowX=dumpyX[0,:]
print("oneRowX: "+str(oneRowX))

newRowX = oneRowX
newRowX[0]=1
newRowX[2]=0
print("newRowX:"+str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY:"+str(predictedY))



