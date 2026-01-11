import numpy as np
import math 
import cv2 as cv
from queue import Queue
from math import acos
import math
# hough 
# 
pi = acos(-1)
def check_border(biniry_image,x,y):
	(x_max,y_max) = biniry_image.shape
	i = x - 1
	count = 0
	while i >= 0:
		if biniry_image[i][y] == 1:
			count += 1
			break
		i -= 1
	i = x + 1
	while i < x_max:
		if biniry_image[i][y] == 1:
			count += 1
			break
		i += 1
	i = y - 1
	while i >= 0:
		if biniry_image[x][i] == 1:
			count += 1
			break
		i -= 1
	i = y + 1
	while i < y_max:
		if biniry_image[x][i] == 1:
			count += 1
			break
		i += 1
	if count == 4:
		return True
	return False

def get_matrix_pad_zeros(matrix):
	(x,y) = matrix.shape
	res = []
	tmp = np.zeros(y+2)
	res.append(tmp)
	for i in range(x):
		tmp = [0]
		for j in range(y):
			tmp.append(matrix[i][j])
		tmp.append(0)
		res.append( np.array(tmp) )
	res.append(np.zeros(y+2))
	return np.array(res)

def get_line_matrix(biniry_image,point,w = 3):
	res_matrix = np.zeros((2*w+1,2*w+1))
	queue = Queue(maxsize = 10000) 
	(x,y) = point
	queue.put(point)
	dx = [-1, -1, -1, 0, 1, 1, 1, 0]
	dy = [-1, 0, 1, 1, 1, 0, -1, -1]

	while queue.empty() != True:
		(get_x,get_y) = queue.get()
		res_matrix[get_x - x + w ][get_y - y + w] = 1
		for i in range(8):
			if biniry_image[get_x + dx[i]][get_y + dy[i]] == 1:

				if abs( get_x + dx[i] - x) <= w and abs(get_y + dy[i]-y) <= w  and res_matrix[get_x + dx[i] - x + w ][get_y+dy[i] - y + w]  == 0:
					new_point = (get_x + dx[i],get_y + dy[i])			
					queue.put(new_point)
	return res_matrix

def norm_l2(point):
	(x,y) = point
	return math.sqrt(x**2 + y**2) 

def scalar_multiplication(point_x , point_y):
	return point_x[0] * point_y[0] + point_x[1] * point_y[1]

def calculation_cos(vector_1, vector_2):
	res = scalar_multiplication(vector_1,vector_2)/(norm_l2(vector_1) * norm_l2(vector_2))
	if res > 1:
		return 1.0
	elif res < -1:
		return -1.0
	else: 
		return res
	
def find_point_of_vetor(matrix, minunatiae_tpye):
	(n,m) = matrix.shape
	if minunatiae_tpye == 0:
		i = 0
		for i in range(m):
			if matrix[0][i] == 1:
				return (0,i)
		for i in range(n):
			if matrix[i][m-1] == 1:
				return (i,m-1)	
		for i in range(m):
			if matrix[n-1][i] == 1:
				return (n-1,i)
		for i in range(n):
			if matrix[i][0] == 1:
				return (i,0)
		return None
	else:
		points = []
		matrix_pad_zeros = get_matrix_pad_zeros(matrix)
		i = 0
		while i < m:
			if matrix[0][i] == 1 and check_minunatiae_at(matrix_pad_zeros,1,i+1) == 0:
				points.append( (0,i) )
			i += 1
		i = 1
		while i < n:
			if matrix[i][m-1] == 1 and check_minunatiae_at(matrix_pad_zeros,i+1,m) == 0:
				points.append(  (i,m-1) )	
			i += 1
		i = m - 2
		while i >  -1:
			if matrix[n-1][i] == 1 and check_minunatiae_at(matrix_pad_zeros,n,i+1) == 0:
				points.append(  (n-1,i) )
			i -= 1
		i = n - 2
		while i >  0: 
			if matrix[i][0] == 1 and check_minunatiae_at(matrix_pad_zeros,i+1,1) == 0:
				points.append( (i,0) )
			i -= 1
		# print(str(n) + ' ' + str(m))
		
		# Check if we have enough points
		if len(points) < 3:
			return None
			
		centroi = (n//2,m//2)
		min_theta = pi
		res_point = None
		vector_1 = ( points[0][0] - centroi[0], points[0][1] - centroi[1] )
		vector_2 = ( points[1][0] - centroi[0], points[1][1] - centroi[1] )
		
		tmp_theta = acos(calculation_cos(vector_1, vector_2))
		if tmp_theta < min_theta: 
			res_point = points[2]
			min_theta = tmp_theta

		# print(matrix)
		vector_1 = ( points[1][0] - centroi[0], points[1][1] - centroi[1] )
		vector_2 = ( points[2][0] - centroi[0], points[2][1] - centroi[1] )
		# print(points[0])
		# print(points[1])
		# print(points[2])
		tmp_theta = acos(calculation_cos(vector_1, vector_2))
		if tmp_theta < min_theta: 
			res_point = points[0]
			min_theta = tmp_theta


		vector_1 = ( points[2][0] - centroi[0], points[2][1] - centroi[1] )
		vector_2 = ( points[0][0] - centroi[0], points[0][1] - centroi[1] )
		tmp_theta = acos(calculation_cos(vector_1, vector_2))
		if tmp_theta < min_theta: 
			res_point = points[0]
			min_theta = tmp_theta

		return res_point


def get_orient(point, minunatiae_tpye, matrix):

	point_two = find_point_of_vetor(matrix,minunatiae_tpye)
	if  point_two == None:
		print('loi tim diem thu 2')
		return None
	(y,x) = point

	(x_centroi,y_centroi) = matrix.shape
	x_centroi = x_centroi // 2 
	y_centroi = y_centroi // 2 
	# print(' ass '+  str (point_two ))
	vector_1 = (0,1)
	vector_2 = (x_centroi - point_two[0], y_centroi - point_two[1])
	# print(vector_2)

	theta = acos(calculation_cos(vector_1, vector_2))
	# print(theta)
	if vector_2[0] >= 0:
		# print(vector_2[0])
		if vector_2[0] == 0:
			if vector_2[1] < 0: 
				return pi
			else:
				return 0.0
		else:
			return 2*pi - theta 
	else:
		return theta 

def check_minunatiae_point(biniry_image,x,y, w = 9):

	(x_max,y_max) = biniry_image.shape
	if x - w < 0 or x + w >= x_max or y - w < 0 or y + w >= y_max:
		return False
	if check_border(biniry_image,x,y) == False:
		return False
	count = 0
	for i in range (-w,w+1):
		for j in range( -w, w+1):
			if i**2 + j** 2 <= 36:
				if check_minunatiae_at(biniry_image, x+ i,y+j) != -1:
					count += 1
	if count > 1:
		return False

	return True

def check_minunatiae_at(biniry_image,x,y):
	
	# for i in range(1 , i_max - 1):
	# 	for j in range(1, j_max - 1):
	# 		print(biniry_image[i][j] , end = ' ')
	# 	print()
	# if biniry_image.shape[0] != 160:
	# 	print(biniry_image)
	dx = [-1, -1, -1, 0, 1, 1, 1, 0, -1]
	dy = [-1, 0, 1, 1, 1, 0, -1, -1,-1]
	if biniry_image[x][y] == 1:
		sum_number = 0
		for k in range(0,8):
			sum_number += abs(biniry_image[ x + dx[k] ][ y + dy[k]] - biniry_image[ x + dx[k+1]][ y + dy[k+1]]  )
		if sum_number//2 == 1:
			return 0
		if sum_number//2 == 3:
			return 1
	return -1

def get_minunatiaes_point(im):
	biniry_image = np.zeros_like(im)
	biniry_image[im<10] = 1.0
	biniry_image = biniry_image.astype(np.int8)
	result = []
	i_max, j_max = biniry_image.shape
	result_im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
	for i in range(2,i_max-1):
		for j in range(2,j_max-1):
			if biniry_image[i][j] == 1:
				minunatiae_tpye = check_minunatiae_at(biniry_image,i,j)
				if minunatiae_tpye != -1 and check_minunatiae_point(biniry_image.copy(),i,j) == True:
					# for k1 in range(-2,3):
					# 	for k2 in range(-2,3):
					# 		print(biniry_image[i+k1][j+k2] , end = " ")
					# 	print()
					#print(str(i) +' '+ str(j))
					#print('')
					matrix = get_line_matrix(biniry_image,(i,j))
					result.append((minunatiae_tpye,(i,j), get_orient((i,j), minunatiae_tpye, matrix )  ) )
					if minunatiae_tpye == 0:
						cv.circle(result_im, (j,i), radius=2, color=(0, 150, 0), thickness=2)
					else:
						cv.circle(result_im, (j,i), radius=2, color=(150, 0, 0), thickness=2)
	
	return result,result_im