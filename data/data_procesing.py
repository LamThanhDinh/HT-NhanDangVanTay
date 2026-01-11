import numpy as np
import glob
import random
import pandas as pd 

def get_name(path):
	name_split = path.split('/')
	return name_split[-1]

def get_information(name):
	information = name.split('_')
	return information[0]
def write_csv(datas,name):
	import csv
	with open(name, mode='w') as csv_file:
		fieldnames = ['path', 'identity']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

		writer.writeheader()
		for row in datas:
			row_writer = {}
			row_writer['path'] = row[0]
			row_writer['identity'] = row[1]
			writer.writerow(row_writer)
	print(f'write {name} done')
def read_csv(path):
	res_datas = []
	pd_csv_reader = pd.read_csv(path) 
	for i in range( len(pd_csv_reader)):
		tmp = []
		tmp.append(pd_csv_reader['path'][i])
		tmp.append(pd_csv_reader['identity'][i])
		res_datas.append(tmp)
	return res_datas

def run():
	path = '/home/dell/Desktop/f/data/dataset/train_data/*.bmp'
	paths = glob.glob(path)
	datas = []
	for row in paths:
		name = get_name(row)
		infors = get_information(name)
		tmp = []
		tmp.append(row)
		tmp.append( int(infors) )
		datas.append(tmp)
	print(len(datas))
	random.shuffle(datas)
	data_choises = datas[:100]
	datas = datas[100:]
	print(len(datas))
	write_csv(datas = data_choises , name = "test_search.csv")
	write_csv(datas = datas , name = "data_create_pretrainmodel.csv")
def split_and_write_datas(datas):
	random.shuffle(datas)
	data_choises = datas[:int(0.33*len(datas))]
	datas = datas[int(0.2*len(datas)):]
	write_csv(datas = data_choises , name = "validation_pretrain_model.csv")
	write_csv(datas = datas , name = "train_pretrain_model.csv")

def main():
	pass


if __name__ == '__main__':
	main()
