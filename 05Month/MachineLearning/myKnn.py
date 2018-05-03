import csv
import random
import math
import operator

def loadDataset(filename, split, trainingset=[], testset=[]):
    with open(filename,'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingset.append(dataset[x])
            else:
                testset.append(dataset[x])
                

def euclideanDistance(instance1, instance2, length):
    distance=0
    for x in range(length):
        distance+=pow(instance1[x]-instance2[x], 2)
    return math.sqrt(distance)

def getNeighbors(trainingset,testInstance, k):
    distances=[]
    length = len(testInstance)-1
    for x in range(len(trainingset)):
        dist = euclideanDistance(testInstance, trainingset[x], length)
        distances.append(trainingset[x],dist)
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
        
def getResponse(neighbors):
    classVotes=[]
    for x in range(len(neighbors)):
        Response=neighbors[x][-1]
        if Response in classVotes:
            classVotes[Response]+=1
        else:
            classVotes[Response]=1
    sortedVotes = sorted(classVotes.iterItems(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]
            
def getAccuracy(testset,predictions):   
    correct=0
    for x in range(len(testset)):
        if testset[x][-1] == predictions[x]:
            correct+=1
    return (correct/float(len(testset)))*100.0

def main():
    trainingSet=[]
    testSet =[]
    split=0.67
    loadDataset(r"play.txt", split, trainingSet, testSet)
    print("train set :"+repr(len(trainingSet)) )
    print("test set :"+repr(len(testSet)))
    
    predictions=[]
    k=3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print("predcted ="+repr(result)+", actual="+repr(testSet[x][-1]))
    accuracy= getAccuracy(testSet, predictions)
    print("accuracy:"+accuracy+"%")
    
main()
    