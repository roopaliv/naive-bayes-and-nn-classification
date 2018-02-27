import pandas as pd
import numpy as np
import math
import sys

def split_data(identifier, dataSet,k):
    if (identifier != '3'):
        dsetSize = len(dataSet)
        ssetSize = int(round(dsetSize/10.0))     
        np.random.shuffle(dataSet)
        for i in range(0,9):
            start = i * ssetSize 
            end = start + ssetSize
            testSet = dataSet[start:end]
            trainSet = np.delete(dataSet,np.s_[start:end],0)
            KNN(trainSet,testSet,k)
        testSet = dataSet[end:]
        trainSet = np.delete(dataSet,np.s_[end:],0)
    else:
        trainSet = dataSet
        testSet = pd.read_table('project3_dataset3_test.txt', delimiter='\t', header = None).values
        testSet= np.insert(testSet, len(testSet[0]), 3, axis = 1)
    return trainSet, testSet

def read_data(filename):
    df = pd.read_table(filename, delimiter='\t', header = None)
    if (int(filename[16]) == 2):
        df[4] = df[4].map({'Present': 1, 'Absent': 0})
    dataSet = df.values
    return dataSet
def get_euclidean_distance(v1,v2):
    sumdis = 0
    for i in range(0,len(v1)-2):
        sumdis = sumdis + math.pow(v1[i] - v2[i],2)
    return math.sqrt(sumdis)

def get_distance_matrix(testSet, trainSet):
    disMat = np.zeros([len(testSet),len(trainSet)])
    for i in range(0,len(testSet)):
        for j in range(0,len(trainSet)):
            disMat[i][j] = get_euclidean_distance(testSet[i],trainSet[j]) 
    return disMat
def get_k_minimum(v,k):               
    temp = np.zeros(k)
    temp2 = np.zeros(k)
    for j in range(0,k):
        minimum = sys.maxint
        index = 0
        for i in range(0,len(v)):
            if v[i] < minimum:
                minimum = v[i]
                index = i
        temp[j] = int(index)
        temp2[j] = minimum
        v[index] = sys.maxint
    for j in range(0,k):
        v[int(temp[j])] = temp2[j]
    return temp

def KNN(trainSet, testSet, k):
    m = len(testSet)
    n = len(trainSet)
    disMat = get_distance_matrix(testSet, trainSet)

    for i in range(0,m):
        nearK = get_k_minimum(disMat[i],k)
        disArray = np.zeros(k)
        classArray = np.zeros(k)
        for j in range(0,k):
            disArray[j]= disMat[i][int(nearK[j])]
            classArray[j] = trainSet[int(nearK[j])][len(trainSet[0])-2]
        weight0 = 0
        weight1 = 0
        for p in range(0,k):
            if classArray[p] == 0:
                temp = math.pow(disArray[p],2)
                weight0 = weight0 + 1 / temp
            elif classArray[p] == 1:
                temp = math.pow(disArray[p],2)
                if temp==0:
                    temp=0.1
                weight1 = weight1 + 1 / temp      
        if weight0 > weight1:
            testSet[i][len(testSet[0])-1] = 0
        elif weight0 < weight1:
            testSet[i][len(testSet[0])-1] = 1


def calc_performance(testSet):
    TP, FN, FP, TN = [float(0)] * 4
    for i in range(0,len(testSet)):
        if testSet[i][len(testSet[0])-2] == 1 and testSet[i][len(testSet[0])-1] == 1:
            TP = TP + 1
        elif testSet[i][len(testSet[0])-2] == 0 and testSet[i][len(testSet[0])-1] == 0:
            TN = TN + 1
        elif testSet[i][len(testSet[0])-2] == 1 and testSet[i][len(testSet[0])-1] == 0:
            FN = FN + 1
        elif testSet[i][len(testSet[0])-2] == 0 and testSet[i][len(testSet[0])-1] == 1:
            FP = FP + 1
    if TP!=0 or b!=0 or c!=0 or d!=0:  
        accuracy = (TP+TN)/(TP+FN+FP+TN)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        fscore = 2*(recall*precision) / (recall+precision)
        print_results(accuracy, precision, recall, fscore)

def print_results(accuracy, precision, recall, fscore):
    print 'Accuracy : '+ str(accuracy)
    print 'Precision : '+  str(precision)
    print 'Recall : '+ str(recall)
    print 'F-measure : '+ str(fscore)            




if __name__ == "__main__":
    files=["project3_dataset1.txt","project3_dataset2.txt","project3_dataset3_train.txt"]
    for filename in files:
        dataSet = read_data(filename)
        dataSet= np.insert(dataSet, len(dataSet[0]), 3, axis = 1)  
        print "########################################"
        print "########################################"
        print "Results for: " + filename
        print "########################################"
        print "########################################"
        k=9
        trainSet, testSet = split_data(filename[16], dataSet,k)
        KNN(trainSet,testSet, k)
        calc_performance(testSet)
        

    

           
                
            
            
            
            
            
        
        
        
            
            
        




