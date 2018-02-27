import numpy as np
import math

def print_results(accuracy, precision, recall, fscore):
	print 'Accuracy : '+ str(accuracy*100) +'%'
	print 'Precision : '+  str(precision*100) +'%'
	print 'Recall : '+ str(recall*100) +'%'
	print 'F-measure : '+ str(fscore*100) +'%' 

def read_data(fileName):
	raw_data, data = open(fileName), []
	record = raw_data.readline()
	while record.strip() != '':
		sample = record.strip().split('\t')
		sample.append(-1)
		data.append(sample)
		record = raw_data.readline()
	raw_data.close()
	for col in range(len(data[0])):
		if convertible(data[0][col]):
			for i in range(len(data)):
				data[i][col] = float(data[i][col])
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

def bayes(data, test_data):
	size, total_features= len(data), len(data[0]) - 2
	total_present, total_absent = [0] * 2
	avg_present, avg_absent, var_present, var_absent= [0]*total_features, [0]*total_features, [0]*total_features, [0]*total_features
	pp, ap, pa, aa = [0]*total_features, [0]*total_features, [0]*total_features, [0]*total_features
	present_rows, absent_rows = [], []
	for row in data:
		if row[-2] == 1:
			present_rows.append(row)
			total_present+=1
			add(avg_present, row[0:-2])
		else:
			absent_rows.append(row)
			total_absent+=1
			add(avg_absent, row[0:-2])
	avg_present[:], avg_absent[:] = [x / total_present for x in avg_present], [x / total_absent for x in avg_absent]
	for i in range(total_features):
		if (convertible(data[0][i])):
			present_attrs, absent_attrs = [], []
			for present_row in present_rows:
				present_attrs.append(present_row[i])
			for absent_row in absent_rows:
				absent_attrs.append(absent_row[i])
			diff = 0
			for row in present_attrs:
				diff += (avg_present[i] - row)**2
			var_present[i] = diff/total_present
			diff = 0
			for row in absent_attrs:
				diff += (avg_absent[i] - row)**2
			var_absent[i] = diff/ total_absent
		else:
			ppi, pai, api, aai= [0]*4
			for absent_row in absent_rows:
				if (absent_row[i] == "Present"):
					pai+=1
				else:
					aai+=1
			for present_row in present_rows:
				if (present_row[i] == "Present"):
					ppi+=1
				else:
					api+=1
			pp[i], ap[i], pa[i], aa[i] = float(ppi)/total_present, float(api)/total_present, float(pai)/total_absent, float(aai)/total_absent
	present, absent = [0.0] *2
	for row in data:
		if row[-2] == 1:
			present+=1
		if row[-2] == 0:
			absent+=1
	if present == 0 or absent == 0:
		present += 1
		absent+=1
		size+=2 
	present /= size
	absent /= size
	for row in test_data:
		features = row[0:total_features]
		attr_present, attr_absent = [0]*total_features, [0]*total_features
		for i in range(total_features):
			currentAttribute = features[i]
			if (convertible(currentAttribute)):
				attr_present[i] = (1/math.sqrt(math.pi*var_present[i]*2))**(.5*(((currentAttribute - avg_present[i])**2)/(var_present[i])))
				attr_absent[i] = (1/math.sqrt(math.pi*var_absent[i]*2))**(.5*(((currentAttribute - avg_absent[i])**2)/(var_absent[i])))
			else: 
				if (currentAttribute == "Present"):
					attr_present[i] = pp[i]
					attr_absent[i] = pa[i]
				else:
					attr_present[i] = ap[i]
					attr_absent[i] = aa[i]
		final_prob_present = reduce(lambda x, y: x*y, attr_present) * present
		final_prob_absent = reduce(lambda x, y: x*y, attr_absent) * absent
		row[-1] = 1 if (final_prob_present > final_prob_absent) else 0

def bayesQuery(data, query, shall_print):
	size, total_features = len(data), len(data[0]) - 2
	present, absent = [0.0] *2
	for row in data:
		if row[-2] == 0:
			absent+=1
		else:
			present+=1
	absent /= size
	present /= size
	attr_present, attr_absent = [0]*total_features, [0]*total_features
	for col in range(total_features):
		p, a = [0.0] *2
		for row in data:
			if (row[col] == query[col]):
				if row[-2] == 0:
					a+=1
				else: 
					p+=1
		attr_present[col] = p/(p+a)
		attr_absent[col] = a/(p+a)
	final_prob_present = reduce(lambda x, y: x*y, attr_present) * present
	final_prob_absent = reduce(lambda x, y: x*y, attr_absent) * absent
	if shall_print:
		print "Results for: " + str(query) 
		print "Prior of a this query to be present: " + str(present*100) + "%"
		print "Prior of a this query to be absent: " + str(absent*100) + "%"
		print "Probability of each attribute to be present: " + str(attr_present)
		print "Probability of each attribute to be absent: " + str(attr_absent)
		print "Probability of X to be present: " + str(final_prob_present)
		print "Probability of X to be absent: " + str(final_prob_absent)
	result = 1 if (final_prob_present > final_prob_absent) else 0
	if shall_print and result == 1:
		print "X will be classified as present"
	elif shall_print and result == 0:
		print "X will be classified as absent"
	return result

def folds_cross_validation(data, identifier):
	folds = 10
	metrics, test_split = [0]*4, len(data)/folds
	test_data, training_data = [[] for x in range(folds)], [[] for x in range(folds)]
	for i in range(folds):
		start = test_split*i
		end = len(data) if (i == (folds-1)) else start + test_split
		test_data[i] = data[start:end]
		training_data[i] = data[0:start] + data[end:len(data)]
		if identifier =='4':
			for testSample in test_data[i]:
				testSample[-1] = bayesQuery(training_data[i], testSample, False)
		else:
			bayes(training_data[i], test_data[i])
		TP, FP, TN, FN = [0.0]*4 
		for row in test_data[i]:
			if row[-2] == 0 and row[-1] == 1:
				FP+=1
			elif row[-2] == 1 and row[-1] == 1:
				TP+=1
			elif row[-2] == 0 and row[-1] == 0:
				TN+=1
			elif row[-2] == 1 and row[-1] == 0:
				FN+=1
		accuracy, recall, precision, fscore = [0]*4
		if TP!=0 or TN!=0 or FP!=0 or FN!=0:  
			accuracy = (TP+TN)/(TP+FN+FP+TN)
			precision = TP / (TP+FP) if (TP!=0 or FP!=0) else 0
			recall = TP / (TP+FN) if (TP!=0 or FN!=0) else 0
			fscore = 2*(recall*precision) / (recall+precision) if (precision!=0 or recall!=0) else 0
		add(metrics, [accuracy, precision, recall, fscore])
	metrics[:] = [x / folds for x in metrics]
	print_results(metrics[0], metrics[1], metrics[2], metrics[3])


def run_results(fileName):
	data = read_data(fileName)
	np.random.seed(3)
	np.random.shuffle(data)
	naive_bayes(fileName[16], data)

def naive_bayes(identifier, data):
	folds_cross_validation(data, identifier)
	if (identifier == '4'):
		run_for_demo_data(data, ["sunny", "cool", "high", "weak"])#["overcast", "cool", "normal", "strong"]

def run_for_demo_data(data, query):
	bayesQuery(data, query, True)

if __name__ == "__main__":
	files=["project3_dataset1.txt","project3_dataset2.txt","project3_dataset4.txt"]
	for fileName in files:
		print "########################################"
		print "########################################"
		print "Results for: " + fileName
		print "########################################"
		print "########################################"
		run_results(fileName)