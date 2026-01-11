import numpy as np 
import torch
from torch.utils import data
import pandas as pd
from .image_procesing import read_image, show_image 

class  image_Dataset(data.Dataset):
	"""docstring for  image_Dataset"""
	def read_csver(self, path):
		infor_images = pd.read_csv(path)
		paths = []
		labels = []
		for i  in range(  len(infor_images)):
			paths.append(infor_images['path'][i])
			labels.append(infor_images['identity'][i])

		return paths,labels

	def __init__(self, path):
		super( image_Dataset, self).__init__()
		self.paths,self.labels = self.read_csver(path = path)
		self.labels = np.array(self.labels)
		self.labels = torch.Tensor(self.labels.astype('long'))

	def __len__(self):
		return len(self.paths) 

	def __getitem__(self,index):
		x = read_image(self.paths[index])
		x = torch.Tensor(x).float()
		y = self.labels[index].long() 
		return x,y
def main():
	dataset = image_Dataset('data_create_pretrainmodel.csv')
	x,y = dataset.__getitem__(1)
	# print(x.shape)

if __name__ == '__main__':
	pass
	# main()


