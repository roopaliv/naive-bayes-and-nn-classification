import pandas as pd
import numpy as np
import math

def read_data(filename):
    df = pd.read_table(filename, delimiter='\t', header = None)
    if (int(filename[16]) == 2):
        df[4] = df[4].map({'Present': 1, 'Absent': 0})
    data = df.values
    return data

def add(l1, l2):
    for index in range(len(l2)):
        l1[index] = l1[index] + l2[index] if convertible(l2[index]) else l1[index]

def convertible(entry):
    try:
        float(entry)
        return True
    except ValueError:
        return False

def run_algo(identifier, data, k):
    metrics = [0]*4
    folds = 10
    split_size = int(round(len(data)/10.0))
    np.random.seed(3)
    np.random.shuffle(data)
    for i in range(folds): # 10 fold cross validation
        if (identifier == '3'):
            training_data = data
            test_data = pd.read_table('project3_dataset3_test.txt', delimiter='\t', header = None).values
            test_data= np.insert(test_data, len(test_data[0]), 3, axis = 1)
            nearest_neighbors_algo(training_data,test_data, k)
            performance_set = test_data
            accuracy, precision, recall, fscore = calc_performance(performance_set)
        else:
            start = split_size*i 
            end = start + split_size
            test_data, training_data = data[start:end], np.delete(data,np.s_[start:end],0)
            nearest_neighbors_algo(training_data,test_data, k)
            performance_set = np.concatenate((training_data, test_data), axis=0)
            #test_data, training_data = data[end:], np.delete(data,np.s_[end:],0)

        accuracy, precision, recall, fscore = calc_performance(performance_set)
        add(metrics, [accuracy, precision, recall, fscore])
    metrics[:] = [x / folds for x in metrics]
    print_results(metrics[0], metrics[1], metrics[2], metrics[3])

def get_k_minimum(neighbor_list,k):               
    neighbor_list_indexes, neighbor_list_values = np.zeros(k), np.zeros(k)
    for j in range(k):
        min_value, min_index = float('inf'), 0
        for i in range(len(neighbor_list)):
            if neighbor_list[i] < min_value:
                min_value, min_index = neighbor_list[i], i
        neighbor_list_values[j], neighbor_list_indexes[j], neighbor_list[min_index]  = min_value, int(min_index), float('inf')
    for j in range(k):
        neighbor_list[int(neighbor_list_indexes[j])] = neighbor_list_values[j]
    return neighbor_list_indexes

def get_euclidean_distance(node1,node2):
    distance = 0
    for i in range(len(node2)-2):
        distance += math.pow(node1[i] - node2[i],2)
    return math.sqrt(distance)

def get_distance_matrix(test_data, training_data):
    distance_matrix = np.zeros([len(test_data),len(training_data)])
    for i in range(len(test_data)):
        for j in range(len(training_data)):
            distance_matrix[i][j] = get_euclidean_distance(test_data[i],training_data[j]) 
    return distance_matrix

def check_neighbors(distance_matrix, test_data, training_data):
    for i in range(len(test_data)):
        k_nearest_neighbors = get_k_minimum(distance_matrix[i],k)
        distances, classes = np.zeros(k), np.zeros(k)
        classify(distances, classes, distance_matrix, test_data, training_data, i, k_nearest_neighbors)

def classify(distances, classes, distance_matrix, test_data, training_data, point, k_nearest_neighbors):
    for j in range(k):
        distances[j]= distance_matrix[point][int(k_nearest_neighbors[j])]
        classes[j] = training_data[int(k_nearest_neighbors[j])][len(training_data[0])-2]
    assign_label_to_point(distances, classes, test_data, point)

def assign_label_to_point(distances, classes, test_data, point):
    probability_false, probability_true = [0]*2
    for l in range(k):
        prob = math.pow(distances[l],2)
        if prob==0:
            prob=0.00001
        prob = 1/prob
        if classes[l] == 0:
            probability_false += prob
        else:
            probability_true += prob      
    if probability_false < probability_true:
        test_data[point][-1] = 1
    else:
        test_data[point][-1] = 0

def nearest_neighbors_algo(training_data, test_data, k):
    distance_matrix = get_distance_matrix(test_data, training_data)
    check_neighbors(distance_matrix, test_data, training_data)

def calc_performance(performance_set):
    TP, FN, FP, TN = [float(0)] * 4
    for i in range(len(performance_set)):
        if performance_set[i][-2] == 0 and performance_set[i][-1] == 0:
            TN += 1
        elif performance_set[i][-2] == 1 and performance_set[i][-1] == 1:
            TP += 1
        elif performance_set[i][-2] == 1 and performance_set[i][-1] == 0:
            FN += 1
        elif performance_set[i][-2] == 0 and performance_set[i][-1] == 1:
            FP += 1
    accuracy, recall, precision, fscore = [0]*4
    if TP!=0 or TN!=0 or FP!=0 or FN!=0:  
        accuracy = (TP+TN)/(TP+FN+FP+TN)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        fscore = 2*(recall*precision) / (recall+precision)
        return accuracy, precision, recall, fscore

def print_results(accuracy, precision, recall, fscore):
    print 'Accuracy : '+ str(accuracy*100) +'%'
    print 'Precision : '+  str(precision*100) +'%'
    print 'Recall : '+ str(recall*100) +'%'
    print 'F-measure : '+ str(fscore*100) +'%'            

if __name__ == "__main__":
    files=["project3_dataset1.txt","project3_dataset2.txt","project3_dataset3_train.txt"]
    k=5
    for filename in files:
        data = read_data(filename)
        data= np.insert(data, len(data[0]), 3, axis = 1)  
        print "########################################"
        print "########################################"
        print "Results for: " + filename
        print "########################################"
        print "########################################"
        run_algo(filename[16], data,k)




