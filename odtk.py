import math
import numpy as np


# how to define theta
# https://github.com/NVIDIA/retinanet-examples/issues/183#issuecomment-617860660
def _corners2rotatedbbox(corners):
    corners = np.array(corners)
    center = np.mean(np.array(corners), 0)
    theta = calc_bearing(corners[0], corners[1])
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    out_points = np.matmul(corners - center, rotation) + center
    x, y = list(out_points[0,:])
    w, h = list(out_points[2, :] - out_points[0, :])
    return [x, y, w, h, theta]

def calc_bearing(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    theta = math.atan2(y2 - y1, x2 - x1)
    theta = nor_theta(theta)
    return theta

def nor_theta(theta):
    if theta > math.radians(45):
        theta -= math.radians(90)
        theta = nor_theta(theta)
    elif theta <= math.radians(-45):
        theta += math.radians(90)
        theta = nor_theta(theta)
    return theta
