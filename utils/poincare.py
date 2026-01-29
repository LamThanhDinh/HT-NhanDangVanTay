from utils import orientation
import math
import cv2 as cv
import numpy as np

def poincare_index_at(i, j, angles, tolerance):
    """
    compute the summation difference between the adjacent orientations such that the orientations is less then 90 degrees
    """
    cells = [(-1, -1), (-1, 0), (-1, 1),         # p1 p2 p3
            (0, 1),  (1, 1),  (1, 0),            # p8    p4
            (1, -1), (0, -1), (-1, -1)]          # p7 p6 p5

    angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells]
    index = 0
    for k in range(0, 8):
        difference = angles_around_index[k] - angles_around_index[k + 1]
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180
        index += difference

    if 180 - tolerance <= index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl"
    return "none"


def calculate_singularities(im, angles, tolerance, W, mask):
    result = cv.cvtColor(im, cv.COLOR_GRAY2BGR) # BGR Consistent

    # BGR Colors
    colors = {
        "loop": (0, 165, 255),    # Orange (Core)
        "delta": (255, 0, 0),      # Blue (Delta)
        "whorl": (255, 0, 255)     # Magenta (Whorl)
    }

    for i in range(3, len(angles) - 2):             # Y
        for j in range(3, len(angles[i]) - 2):      # x
            mask_slice = mask[(i-2)*W:(i+3)*W, (j-2)*W:(j+3)*W]
            mask_flag = np.sum(mask_slice)
            if mask_flag == (W*5)**2:
                singularity = poincare_index_at(i, j, angles, tolerance)
                if singularity != "none":
                    center_x = (j * W) + (W // 2)
                    center_y = (i * W) + (W // 2)
                    box_size = W + 2
                    # Màu: cam (core), đỏ (delta), tím (whorl)
                    color = (0, 165, 255) if singularity == "loop" else ((0, 0, 255) if singularity == "delta" else (255, 0, 255))
                    thickness = 2 if singularity == "delta" else 2
                    # Vẽ ô vuông (rectangle) tại vị trí singularity
                    top_left = (center_x - box_size//2, center_y - box_size//2)
                    bottom_right = (center_x + box_size//2, center_y + box_size//2)
                    cv.rectangle(result, top_left, bottom_right, color, thickness)


    return result

if __name__ == '__main__':
    img = cv.imread('../test_img.png', 0)
    # Testing code...
