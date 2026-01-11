import numpy as np
import cv2 as cv
import math


def create_segmented_and_variance_images(im, w, threshold=.3):
    """
    Trả về mặt identifying ROI. Tính độ lệch chuẩn trong từng khối hình ảnh và ngưỡng ROI
    Nó cũng bình thường hóa các giá trị intesity của hình ảnh sao cho các vùng sườn núi có giá trị trung bình bằng 0, đơn vị độ chuẩn
    sai lệch.
    :param im: Image
    :param w: kích cỡ của block
    :param threshold: std ngưỡng
    :return: segmented_image
    """
    (y, x) = im.shape
    threshold = np.std(im)*threshold

    image_variance = np.zeros(im.shape)
    segmented_image = im.copy()
    mask = np.ones_like(im)
    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)]
            block_stddev = np.std(im[box[1]:box[3], box[0]:box[2]])
            image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev

    # loc theo nguong threshold
    mask[image_variance < threshold] = 0
    # lam muot anh
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(w*2, w*2))
    # for i in range(32):
    #     for j in range(32):
    #         print(kernel[i][j], end=' ')
    #     print()
    # xóa nhiễu muối tiêu
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # điền đầy các khoảng trống
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # xóa nền
    segmented_image *= mask
    # chuẩn hóa lại ảnh sau khi xóa nền
    mean_val = np.mean(im[mask==0]) # trung bình mức xám vùng nền
    std_val = np.std(im[mask==0])# độ lệch chuẩn mức xám vùng nền
    if  math.isnan(mean_val):
        mean_val = 0
        std_val = 1
    # làm cho các giá trị mức xám ở vùng nền đồng đều nhau hơn
    norm_img = (im - mean_val)/(std_val)
    a_min = np.amin(norm_img)
    norm_img -= a_min
    return segmented_image, norm_img, mask