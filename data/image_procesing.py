
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import pi, cos, sin

def read_image(path):
	x = cv.imread(path,cv.IMREAD_GRAYSCALE)
	return x
def read_image_rgb(path):
	image = cv.imread(path,cv.IMREAD_GRAYSCALE)
	#image = cv.resize(image,(96,96),interpolation = cv.INTER_AREA)
	return image

def read_images(paths):
	datas_return = []
	p = {}
	for path in paths:
		image = cv.imread(path,cv.IMREAD_GRAYSCALE)
		datas_return.append(image) 
		if str(image.shape) not in p:
			p[str(image.shape)] = 1
		else:
			p[str(image.shape)] += 1
	print(p)
	return np.array( datas_return )
def equalize_historgram(image):
	return cv.equalizeHist(image)
# infor_images = pd.read_csv('data_create_pretrainmodel.csv')
# read_images(infor_images['path'])
def show_image(image, title):
  # if len(image.shape) < 3: 
  #   image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
  plt.figure()
  plt.title(title)
  plt.imshow(image)
  plt.show()

'''ig = read_image_rgb('/home/dell/Desktop/f/data/dataset/real_data/00009.bmp')
show_image(ig,'resize')
ig = equalize_historgram(ig)
show_image(ig,'resize')
'''
def rest_point(point):
    b = 0
    x = point[0]
    y = point[1]
    theta = point[2] + pi/2
    k = math.tan(theta)
    if(theta == pi/2 or theta == 3*pi/2):
      return [x, y + 10]
    elif(theta == 0 or theta == pi):
      return [x+10, y]
    elif(x == 0):
      b = y
    else:
      b = float(y - k*x)
    x1 = x+10
    y1 = round(x1*k+b)
    return [x1, y1]

def rest_point1(point):
    x1 = point[0]
    y1 = point[1]
    theta = point[2] + pi/2
    x2 = 10*cos(theta)
    y2 = 10*sin(theta)
    x2 += x1
    y2 += y1
    return [round(x2), round(y2)]
