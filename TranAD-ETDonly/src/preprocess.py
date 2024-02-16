import os
import pandas as pd
import numpy as np
import sys

import json


datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', 'ETD']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']


def load_and_save(category, filename, dataset, dataset_folder, output_folder):
	temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
						dtype=np.float64,
						delimiter=',')
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
	return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape, output_folder):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset, output_folder='./processed', data_folder='./data/ETD'):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	file = os.path.join(data_folder, 'ETTh1.csv')
	df = pd.read_csv(file)
	df.drop(['date'], axis=1, inplace=True)
	length = df.shape[0]
	df_train = df.iloc[0:int(0.8*length)]
	df_test = df.iloc[int(0.8*length):]
	train = []
	test = []
	for category in df_train.columns:
		train1, min_1, max_1 = normalize2(df_train[category].values)
		test1, _, _ = normalize2(df_test[category].values, min_1, max_1)
		# print(train1.shape)
		train.append(train1)
		test.append(test1)
	train.append(np.zeros((len(train[0]),)))
	test.append(np.zeros((len(test[0]),)))
	train = np.array(train)
	test = np.array(test)
	#print(train.shape)
	# train, min_a, max_a = normalize2(df_train.values)
	# test, _, _ = normalize2(df_test.values, min_a, max_a)
	train = train.transpose((1, 0))
	test = test.transpose((1, 0))
	label = np.zeros((test.shape[0], test.shape[1]))
	np.save(
		os.path.join(folder, f"train.npy"),
		train,
	)
	np.save(
		os.path.join(folder, f"test.npy"),
		test,
	)
	np.save(
		os.path.join(folder, f"labels.npy"),
		label,
	)

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")