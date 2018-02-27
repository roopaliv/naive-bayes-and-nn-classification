
import pandas as pd
import numpy as np

def preprocess_data(file_path):
	df = pd.read_table(file_path, delimiter='\t', header = None)
	df[4] = df[4].map({'Present': 1, 'Absent': 0})
	#df.columns.append(len(df.columns)+1)
	df= np.insert(df, len(df[0]), 5, axis = 1)
	
	return df

if __name__ == "__main__":
	file_path = 'project3_dataset2.txt'
	df = pd.read_table(file_path, delimiter='\t', header = None)
	df[4] = df[4].map({'Present': 1, 'Absent': 0})
	print df