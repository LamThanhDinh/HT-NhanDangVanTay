import cv2 as cv
import numpy as np


def minutiae_at(pixels, i, j, kernel_size):
    """
    https://airccj.org/CSCP/vol7/csit76809.pdf pg93
    Crossing number methods is a really simple way to detect ridge endings and ridge bifurcations.
    Then the crossing number algorithm will look at 3x3 pixel blocks:

    if middle pixel is black (represents ridge):
    if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
    if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation

    :param pixels:
    :param i:
    :param j:
    :return:
    """
    # if middle pixel is black (represents ridge)
    if pixels[i][j] == 1:

        if kernel_size == 3:
            cells = [(-1, -1), (-1, 0), (-1, 1),        # p1 p2 p3
                   (0, 1),  (1, 1),  (1, 0),            # p8    p4
                  (1, -1), (0, -1), (-1, -1)]           # p7 p6 p5
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),                 # p1 p2   p3
                   (-1, 2), (0, 2),  (1, 2),  (2, 2), (2, 1), (2, 0),               # p8      p4
                  (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]           # p7 p6   p5

        values = [pixels[i + l][j + k] for k, l in cells]

        # count crossing how many times it goes from 0 to 1
        crossings = 0
        for k in range(0, len(values)-1):
            crossings += abs(values[k] - values[k + 1])
        crossings //= 2

        # if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
        # if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"

    return "none"


def calculate_minutiaes(im, kernel_size=3):
    biniry_image = np.zeros_like(im)
    biniry_image[im<10] = 1.0
    biniry_image = biniry_image.astype(np.int8)

    (y, x) = im.shape
    result = cv.cvtColor(im, cv.COLOR_GRAY2BGR) # Use BGR for OpenCV
    # RED for Ending, GREEN for Bifurcation (BGR format)
    colors = {"ending" : (0, 0, 255), "bifurcation" : (0, 255, 0)}

    # iterate each pixel minutia
    for i in range(1, x - kernel_size//2):
        for j in range(1, y - kernel_size//2):
            minutiae = minutiae_at(biniry_image, j, i, kernel_size)
            if minutiae != "none":
                # Draw filled circle with small radius
                cv.circle(result, (i,j), radius=3, color=colors[minutiae], thickness=-1)
                # Optional: Add white contour for better visibility
                # cv.circle(result, (i,j), radius=3, color=(255,255,255), thickness=1)

    # --- ADD LEGEND ON IMAGE ---
    # Box background (White)
    legend_w = 110
    legend_h = 50
    margin = 10
    box_top_left = (margin, y - legend_h - margin)
    box_bottom_right = (margin + legend_w, y - margin)
    
    # Draw semi-transparent background if possible, but for simplicity solid white
    cv.rectangle(result, box_top_left, box_bottom_right, (255, 255, 255), -1)
    cv.rectangle(result, box_top_left, box_bottom_right, (0, 0, 0), 1) # Black border

    # Ending Item
    cv.circle(result, (margin + 15, y - margin - 35), 4, colors["ending"], -1)
    cv.putText(result, "Ending", (margin + 30, y - margin - 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)

    # Bifurcation Item
    cv.circle(result, (margin + 15, y - margin - 15), 4, colors["bifurcation"], -1)
    cv.putText(result, "Bifurcation", (margin + 30, y - margin - 10), 
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)

    return result
