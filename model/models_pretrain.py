
'''
import library 
'''
from torchsummary import summary
import torch 
from torch import nn
import numpy as np 
import torch.nn.functional as F
class CNNnet(torch.nn.Module):
	"""docstring for CNNnet"""
	def __init__(self):
		super(CNNnet,self).__init__()
		# input shape = (1,160,160)
		self.conv2d1 = nn.Sequential(
				nn.Conv2d(1,4,kernel_size = 3, stride = 1, padding = 0),
				#output shape = (4,158,158)
				nn.BatchNorm2d(4),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = 2, stride = 2)
			)
		#output shape = (4,79,79)
		# input shape = (4,79,79)
		self.conv2d2 = nn.Sequential(
				nn.Conv2d(4,16,kernel_size = 3, stride = 1, padding = 1),
				#output shape = (8,49,46)
				nn.BatchNorm2d(16),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = 2, stride = 2)
			)
		#output shape = (8,24,23)
		self.conv2d3 = nn.Sequential(
				nn.Conv2d(16,32,kernel_size = 3, stride = 1, padding = 0),
				#output shape = (16,22,21)
				nn.BatchNorm2d(32),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = 2, stride = 2)
			)
		#output shape = (16,11,10)
	
		self.conv2d4 = nn.Sequential(
				nn.Conv2d(32,64,kernel_size = 3, stride = 1, padding = 0),
				#output shape = (8,9,8)
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = 2, stride = 2)
		)
		#output shape = (8,4,4)
		self.fc1 = nn.Linear(64*8*8,256)
		self.dp1 = nn.Dropout(0.2)
		self.fc2 = nn.Linear(256,64)
		self.dp2 = nn.Dropout(0.5)
		self.fc3 = nn.Linear(64,10)


	def forward(self,x):

		out = self.conv2d1(x)
		out = self.conv2d2(out)
		out = self.conv2d3(out)
		out = self.conv2d4(out)
		out = out.view(-1,64*8*8)
		out = self.fc1(out)
		out = self.dp1(out)
		out = self.fc2(out)
		out = self.dp2(out)
		out = self.fc3(out)
		return out
'''
def summary_model():		
	model = CNNnet()
	print(summary( model ,(1,160,160)))

summary_model()
'''
		

		


