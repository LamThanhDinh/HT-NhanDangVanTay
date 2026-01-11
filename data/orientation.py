import math
import numpy as np
import cv2 as cv



def calculate_angles(im, W):
    """
    :param im:
    :param W: int width of the ridge
    :return: array
    """
    j1 = lambda x, y: 2 * x * y
    j2 = lambda x, y: x ** 2 - y ** 2
    j3 = lambda x, y: x ** 2 + y ** 2

    (y, x) = im.shape

    sobelOperator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ySobel = np.array(sobelOperator).astype(int)
    xSobel = np.transpose(ySobel).astype(int)

    result = [[] for i in range(1, y, W)]
    max_value = np.amax(im)
    Gx_ = cv.filter2D(im/max_value,-1, ySobel)*max_value
    Gy_ = cv.filter2D(im/max_value,-1, xSobel)*max_value
    for j in range(1, y, W):
        for i in range(1, x, W):
            nominator = 0
            denominator = 0
            for l in range(j, min(j + W, y - 1)):
                for k in range(i, min(i + W , x - 1)):
                    Gx = round(Gx_[l, k])  # Tính gradient theo Ox tại (l, k)
                    Gy = round(Gy_[l, k])  # Tính gradient theo Oy tại (l, k)
                    nominator += j1(Gx, Gy)
                    denominator += j2(Gx, Gy)

            if nominator or denominator:
                angle = (math.pi + math.atan2(nominator, denominator)) / 2
                # orientation = np.pi/2 + math.atan2(nominator,denominator)/2
                result[int((j-1) // W)].append(angle)
            else:
                result[int((j-1) // W)].append(0)
                # print(nominator)
                # print(denominator)
                
    result = np.array(result)
    return result

def get_line_ends(i, j, W, tang):
    if -1 <= tang and tang <= 1:
        begin = (i, int((-W/2) * tang + j + W/2))
        end = (i + W, int((W/2) * tang + j + W/2))
    else:
        begin = (int(i + W/2 + W/(2 * tang)), j + W//2)
        end = (int(i + W/2 - W/(2 * tang)), j - W//2)
    return (begin, end)

def visualize_angles(im, mask, angles, W):
    (y, x) = im.shape
    result = cv.cvtColor(np.zeros(im.shape, np.uint8), cv.COLOR_GRAY2RGB)
    mask_threshold = (W-1)**2
    for i in range(1, x, W):
        for j in range(1, y, W):
            radian = np.sum(mask[j - 1:j + W, i-1:i+W])
            if radian > mask_threshold:
                tang = math.tan(angles[(j - 1) // W][(i - 1) // W])
                (begin, end) = get_line_ends(i, j, W, tang)
                cv.line(result, begin, end, color=150)

    cv.resize(result, im.shape, result)
    return result