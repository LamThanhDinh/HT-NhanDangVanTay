import math
from math import pi
from math import sqrt
from math import cos, sin
import numpy as np
import json

def calculate_distance(vector1, vector2):
    # Check if orientation values are None
    if vector1[3] is None or vector2[3] is None:
        return (float('inf'), float('inf'))
    
    sd = sqrt((vector1[1]-vector2[1])*(vector1[1]-vector2[1]) + (vector1[2]-vector2[2])*(vector1[2]-vector2[2]))
    dd = min(abs(vector1[3]-vector2[3]), 2*pi-abs(vector1[3]-vector2[3]))
    return (sd, dd)

# def hough_transform(I, T, theta=5, err_theta=1, err_x=5, err_y=5, step_theta=5):
#     delta_theta = np.arange(0, 360, step_theta)
#     A = []
#     for mt in T:
#         for mi in I:
#             for t in delta_theta:
#                 tt = abs(180*mi[3]/pi + t - 180*mt[3]/pi)
#                 dd = min(tt, 360-tt)
#                 if(dd < theta):
#                     mt_matrix = np.array([[mt[1]], [mt[2]]])
#                     mi_matrix = np.array([[mi[1]], [mi[2]]])
#                     rotation = np.array([[cos(t*pi/180), sin(t*pi/180)*(-1)], [sin(t*pi/180), cos(t*pi/180)]])
#                     delta_x_y = mt_matrix - rotation.dot(mi_matrix)
#                     delta_x = np.arange(delta_x_y[0]-err_x, delta_x_y[0]+err_x)
#                     delta_y = np.arange(delta_x_y[1]-err_y, delta_x_y[1]+err_y)
#                     delta_t = np.arange(t-err_theta, t+err_theta)
#                     for i in delta_t:
#                         for j in delta_x:
#                             for k in delta_y:
#                                 if(len(A) == 0):
#                                     A.append({
#                                         'set': [round(j), round(k), i],
#                                         'count': 1
#                                     })
#                                 else:
#                                     for a in A:
#                                         if(a['set'] == [round(j), round(k), i]):
#                                             a['count'] += 1
#                                             break
#                                         else:
#                                             A.append({
#                                                 'set': [round(j), round(k), i],
#                                                 'count': 1
#                                             })
#                                             break


#     max_vote = 0
#     result_set = []
#     for a in A:
#         if(a['count'] > max_vote):
#             result_set = a['set']
#             max_vote = a['count']

#     return (result_set, max_vote)

# def rotate_image(mi, delta_x, delta_y, delta_t):
#     mi_matrix = np.array([[mi[1]], [mi[2]]])
#     rotation = np.array([[cos(delta_t), sin(delta_t) * (-1)], [sin(delta_t), cos(delta_t)]])
#     delta_x_y_matrix = np.array([[delta_x], [delta_y]])
#     result_matrix = rotation.dot(mi_matrix) + delta_x_y_matrix
#     rel = []
#     rel.append(mi[0])
#     rel.append(round(list(result_matrix[0])[0]))
#     rel.append(round(list(result_matrix[1])[0]))
#     if(mi[3] + delta_t > 2*pi):
#         rel.append(mi[3] + delta_t - 2*pi)
#     else: rel.append(mi[3] + delta_t)
#     return rel

# def count_minuntiae_matching(I, T, delta_x, delta_y, delta_t, r=10, theta=pi/24):
#     count = 0
#     check = np.zeros(len(T))
#     for mi in I:
#         for t, mt in enumerate(T):
#             if(check[t] == 0):
#                 rotated_minuntiae = rotate_image(mi, delta_x, delta_y, (pi*delta_t)/180)
#                 (sd, dd) = calculate_distance(rotated_minuntiae, mt)
#                 if(sd <= r and dd <= theta):
#                     count += 1
#                     check[t] = 1
#     return count

# def count_brute_force(I, T, r0, t0):
#     count = 0
#     check = np.zeros(len(T))
#     for mi in I:
#         for i, mt in enumerate(T):
#             if(check[i] == 0):
#             # if(check[i] == 0 and mi[0] == mt[0]):
#                 (sd, dd) = calculate_distance(mi, mt)
#                 if(sd <= r0 and dd <= t0):
#                     count += 1
#                     check[i] = 1
#     return count

# def brute_force(I, data, r0, t0):
#     max_matching_point = 0
#     index_matching = 0
#     for i in data:
#         if i['points'] == I:
#             continue
#         count = count_brute_force(I, i['points'], r0, t0)
#         if (count > max_matching_point):
#             max_matching_point = count
#             index_matching = i
#     return (index_matching, max_matching_point)



