import numpy as np
import math
import scipy.ndimage
from .image_procesing import *
def inmatrix(matrix):
    for i in matrix:
        for j in i:
            print(j,end=' ')
    print()

def frequest(im, orientim, kernel_size, minWaveLength, maxWaveLength):
    """
    """
    rows, cols = np.shape(im)
    cosorient = np.cos(2*orientim) # np.mean(np.cos(2*orientim))
    sinorient = np.sin(2*orientim) # np.mean(np.sin(2*orientim))
    block_orient = math.atan2(sinorient,cosorient)/2
    # xoay ảnh sao đường vân trùng với trục oy
    rotim = scipy.ndimage.rotate(im,block_orient/np.pi*180 - 90,axes=(1,0),reshape = False,mode = 'nearest')
    # show_image(im,'truoc')~
    # show_image(rotim,'sau')
    # chỉ lấy vùng vân nơi mà đường vân đủ độ tin tưởng là 1 vân chứ không phải island
    cropsze = int(np.fix(rows/np.sqrt(2)))
    offset = int(np.fix((rows-cropsze)/2))
    rotim = rotim[offset:offset+cropsze][:,offset:offset+cropsze]
    # show_image(rotim,'sau')
    # tính tổng giá mức xám dọc theo đường vân
    ridge_sum = np.sum(rotim, axis = 1)
    # inmatrix(rotim)
    #     # ix = input()
    dilation = scipy.ndimage.grey_dilation(ridge_sum, kernel_size, structure=np.ones(kernel_size))
    ridge_noise = np.abs(dilation - ridge_sum)
    peak_thresh = 2
    maxpts = (ridge_noise < peak_thresh) & (ridge_sum > np.mean(ridge_sum))
    maxind = np.where(maxpts)
    _, no_of_peaks = np.shape(maxind)
    
    if(no_of_peaks<2):
        freq_block = np.zeros(im.shape)
    else:
        waveLength = (maxind[0][-1] - maxind[0][0])/(no_of_peaks - 1)
        if waveLength>=minWaveLength and waveLength<=maxWaveLength:
            freq_block = 1/np.double(waveLength) * np.ones(im.shape)
        else:
            freq_block = np.zeros(im.shape)
    return(freq_block)


def ridge_freq(im, mask, orient, block_size, kernel_size, minWaveLength, maxWaveLength):
    # Ước lượng tần suất đường vân
    rows,cols = im.shape
    freq = np.zeros((rows,cols))

    for row in range(0, rows - block_size, block_size):
        for col in range(0, cols - block_size, block_size):
            image_block = im[row:row + block_size][:, col:col + block_size]
            angle_block = orient[row // block_size][col // block_size]
            if angle_block:
                freq[row:row + block_size][:, col:col + block_size] = frequest(image_block, angle_block, kernel_size,
                                                                               minWaveLength, maxWaveLength)
    freq = freq*mask
    freq_1d = np.reshape(freq,(1,rows*cols))
    ind = np.where(freq_1d>0)
    ind = np.array(ind)
    ind = ind[1,:]
    non_zero_elems_in_freq = freq_1d[0][ind]
    medianfreq = np.median(non_zero_elems_in_freq) * mask

    return medianfreq