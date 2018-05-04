'''
Created on Dec 29, 2015

@author: Alex
'''
import math
from numba import jit, float32
import numpy as np


def determine_longer_and_shorter_lines(line_a, line_b):
    if line_a.length < line_b.length:
        return (line_b, line_a)
    else:
        return (line_a, line_b)


def get_total_distance_function(perp_dist_func, angle_dist_func, parrallel_dist_func):
    def __dist_func(line_a, line_b, perp_func=perp_dist_func, angle_func=angle_dist_func, \
                    parr_func=parrallel_dist_func):
        return perp_func(line_a, line_b) + angle_func(line_a, line_b) + \
            parr_func(line_a, line_b)
    return __dist_func

@jit(float32(float32, float32))
def perpendicular_distance_res(dist_a, dist_b):
    return (dist_a * dist_a + dist_b * dist_b) / (dist_a + dist_b)

def perpendicular_distance2(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)
    dist_a = shorter_line.start.distance_to_projection_on2(longer_line)
    dist_b = shorter_line.end.distance_to_projection_on2(longer_line)

    if dist_a == 0.0 and dist_b == 0.0:
        return 0.0

    return perpendicular_distance_res(dist_a, dist_b)

@jit(float32(float32, float32))
def perpendicular_distance_numba(da, db):
    if da == 0.0 and db == 0.0:
        return 0.0
    return (da * da + db * db) / (da + db)

def perpendicular_distance(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)
    dist_a = shorter_line.start.distance_to_projection_on(longer_line)
    dist_b = shorter_line.end.distance_to_projection_on(longer_line)

    # return perpendicular_distance_numba(dist_a, dist_b)
    if dist_a == 0.0 and dist_b == 0.0:
        return 0.0

    return (dist_a * dist_a + dist_b * dist_b) / (dist_a + dist_b)


def __perpendicular_distance(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)
    dist_a = longer_line.line.project(shorter_line.start).distance_to(shorter_line.start)
    dist_b = longer_line.line.project(shorter_line.end).distance_to(shorter_line.end)

    if dist_a == 0.0 and dist_b == 0.0:
        return 0.0
    else:
        return (math.pow(dist_a, 2) + math.pow(dist_b, 2)) / (dist_a + dist_b)

# @jit
@jit(float32(float32, float32, float32))
def angular_distance_res(angle1, angle2, length):
    t = angle1 - angle2
    if (t < 90 and t > 0) or (t < 360 and t > 270):
        return abs(np.sin(t * np.pi / 180)) * length
    else:
        return length

def angular_distance3(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)
    if line_a.angle == None or line_b.angle == None:
        sine_coefficient = shorter_line.sine_of_angle_with(longer_line)
        return abs(sine_coefficient * shorter_line.length)
    else:
        return angular_distance_res(line_a.angle, line_b.angle, longer_line.length)

def angular_distance2(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)
    if line_a.angle == None or line_b.angle == None:
        sine_coefficient = shorter_line.sine_of_angle_with(longer_line)
        return abs(sine_coefficient * shorter_line.length)
    else:
        # t = (line_a.angle + line_b.angle)/2.0 * np.pi / 180
        # return abs(np.sin((t - np.pi)/2) + 1) * shorter_line.length
        t = abs(line_a.angle - line_b.angle)
        if (t < 30 and t > 0) or (t < 360 and t > 330):
            return abs(np.sin(t * np.pi / 180)) * shorter_line.length
        else:
            return longer_line.length*100000
            # return longer_line.length*100000



def angular_distance(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)
    if ((line_a.end.y - line_a.start.y)==0 or (line_b.end.y - line_b.start.y)==0) and (((line_a.end.x - line_a.start.x)*(line_b.end.x - line_b.start.x))<0):
        #如果两条线水平平行，比较x的方向是否相同，如果不相同说明是钝角
        return longer_line.length*10000
    elif((line_a.end.x - line_a.start.x)==0 or (line_b.end.x - line_b.start.x)==0) and (((line_a.end.y - line_a.start.y)*(line_b.end.y - line_b.start.y))<0):
        # 如果两条线垂直平行，比较y的方向是否相同，如果不相同说明是钝角
        return longer_line.length*10000
    elif(((line_a.end.y - line_a.start.y)*(line_a.end.x - line_a.start.x)*(line_b.end.y - line_b.start.y)*(line_b.end.x - line_b.start.x))<0):
        #判断是否为钝角
        return longer_line.length*10000
    else:
        #夹角为锐角
        sine_coefficient = shorter_line.sine_of_angle_with(longer_line)
        return abs(sine_coefficient * shorter_line.length)


def  velocity_distance(line_a, line_b):
    return abs(line_a.v - line_b.v)*1800
# def __parrallel_distance(line_a, line_b):

# @jit


def parrallel_distance(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)

    def __func(shorter_line_pt, longer_line_pt):
        return shorter_line_pt.distance_from_point_to_projection_on_line_seg(longer_line_pt, \
                                                                             longer_line)
    return min([longer_line.dist_from_start_to_projection_of(shorter_line.start), \
                longer_line.dist_from_start_to_projection_of(shorter_line.end), \
                longer_line.dist_from_end_to_projection_of(shorter_line.start), \
                longer_line.dist_from_end_to_projection_of(shorter_line.end)])


def dist_to_projection_point(line, proj):
    return min(proj.distance_to(line.start), proj.distance_to(line.end))
