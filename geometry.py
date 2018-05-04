'''
Created on Mar 23, 2016

@author: Alex
'''
import math
import numba
import datetime
from traclus_impl.representative_trajectory_average_inputs import DECIMAL_MAX_DIFF_FOR_EQUALITY
from dateutil import parser


_delta = 0.000000001

def set_max_delta_for_equality(delta):
    _delta = delta

class Vec2(object):
    def __init__(self, x, y, C=None, v = None, TIME = None):
        self.x = float(x)
        self.y = float(y)

        if C!=None:
            self.c = float(C)
        else:
            self.c = None
        if v!=None:
            self.v = float(v)
        else:
            self.v = None

        if TIME!= None:
            self.t = parser.parse(TIME)
        else:
            self.t = None

        if x != 0.0:
            self.angle = math.degrees(math.atan(float(y) / x))
        elif y == 0.0:
            self.angle = 0
        elif y > 0.0:
            self.angle = 90
        elif y < 0.0:
            self.angle = -90
            
        if self.x < 0:
            self.angle += 180
        
    def dot_product_with(self, other_vector):
        return self.x * other_vector.x + self.y * other_vector.y
    
    def as_dict(self):
        return {'x': self.x, 'y': self.y, 'c': self.c, 'v': self.v, 'time': self.t.strftime("%Y-%m-%d %H:%M:%S")}
    
    def multipled_by_matrix(self, x1, y1, x2, y2):
        new_x = self.x * x1 + self.y * x2
        new_y = self.x * y1 + self.y * y2
        return Vec2(new_x, new_y)
    
    def rotated(self, angle_in_degrees):
        cos_angle = math.cos(math.radians(angle_in_degrees))
        sin_angle = math.sin(math.radians(angle_in_degrees))
        return self.multipled_by_matrix(x1=cos_angle, y1=sin_angle, x2=-sin_angle, y2=cos_angle)
    
    def almost_equals(self, other):
        return abs(self.x - other.x) <= DECIMAL_MAX_DIFF_FOR_EQUALITY and \
            abs(self.y - other.y) <= DECIMAL_MAX_DIFF_FOR_EQUALITY
            
    def __eq__(self, other):
        return other != None and abs(self.x - other.x) < _delta and abs(self.y - other.y) < _delta
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __str__(self):
        return "x: " + str(self.x) + ". y: " + str(self.y)
    
def distance(diff_x, diff_y):
    return math.sqrt(diff_x * diff_x + diff_y * diff_y)

@numba.jit(numba.float32(numba.float32, numba.float32, numba.float32, numba.float32, numba.float32, numba.float32))
def distance_to_projection_func(x, y, start_x, start_y, unit_vector_x, unit_vector_y):
    diff_x = x - start_x
    diff_y = y - start_y
    return abs(diff_x * unit_vector_y - diff_y * unit_vector_x)


class Point(Vec2):
    def __init__(self, x, y, C=None, V = None, TIME = None):
        Vec2.__init__(self, x, y, C, V, TIME)
        
    def distance_to(self, other_point):
        diff_x = other_point.x - self.x
        diff_y = other_point.y - self.y
        return math.sqrt(math.pow(diff_x, 2) + math.pow(diff_y, 2))

    def distance_to_projection_on2(self, line_segment):
        return distance_to_projection_func(self.x, self.y, line_segment.start.x, line_segment.start.y, line_segment.unit_vector.x, line_segment.unit_vector.y)

    @staticmethod
    @numba.jit(numba.float32(numba.float32, numba.float32, numba.float32, numba.float32, numba.float32, numba.float32))
    def distance_to_projection_on_numba(x, y, start_x, start_y, unit_vector_x, unit_vector_y):
        diff_x = x - start_x
        diff_y = y - start_y
        return abs(diff_x * unit_vector_y - diff_y * unit_vector_x)

    def distance_to_projection_on(self, line_segment):
        # return self.distance_to_projection_on_numba(self.x, self.y, line_segment.start.x, line_segment.start.y, line_segment.unit_vector.x, line_segment.unit_vector.y)
        diff_x = self.x - line_segment.start.x
        diff_y = self.y - line_segment.start.y

        return abs(diff_x * line_segment.unit_vector.y - diff_y * line_segment.unit_vector.x)
    
    def rotated(self, angle_in_degrees):
        result = Vec2.rotated(self, angle_in_degrees)
        return Point(result.x, result.y)
        
class LineSegment(object):
    @staticmethod
    def from_tuples(start, end):
        return LineSegment(Point(start[0], start[1]), Point(end[0], end[1]))
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.length = start.distance_to(end)
        self.v = (start.v + end.v)/2.
        if start.c !=None and end.c != None:
            self.angle = (end.c + start.c)/2.0
        else:
            self.angle = None
        if self.length > 0.0:
            unit_x = (end.x - start.x) / self.length
            unit_y = (end.y - start.y) / self.length
            self.unit_vector = Point(unit_x, unit_y)
        else:
            unit_x = 0
            unit_y = 0
            self.unit_vector = Point(unit_x, unit_y)
            
    def as_dict(self):
        return {'start': self.start.as_dict(), 'end': self.end.as_dict(), 'angle': self.angle}

    @staticmethod
    @numba.jit(numba.float32(numba.float32, numba.float32, numba.float32, numba.float32))
    def sine_of_angle_with_numba(x1, y1, x2, y2):
        return x1*y2 - y1*x2
        
    def sine_of_angle_with(self, other_line_segment):
        # return self.sine_of_angle_with_numba(self.unit_vector.x, self.unit_vector.y, other_line_segment.unit_vector.x, other_line_segment.unit_vector.y)
        return self.unit_vector.x * other_line_segment.unit_vector.y - \
        self.unit_vector.y * other_line_segment.unit_vector.x

    @staticmethod
    @numba.jit(numba.float32(numba.float32, numba.float32, numba.float32, numba.float32, numba.float32, numba.float32))
    def dist_from_start_to_projection_of_numba(x, y, start_x, start_y, unit_vector_x, unit_vector_y):
        diff_x = start_x - x
        diff_y = start_y - y
        return abs(diff_x * unit_vector_x + diff_y * unit_vector_y)


    def dist_from_start_to_projection_of(self, point):
        # return self.dist_from_start_to_projection_of_numba(point.x, point.y, self.start.x, self.start.y, self.unit_vector.x, self.unit_vector.y)
        diff_x = self.start.x - point.x
        diff_y = self.start.y - point.y
        return abs(diff_x * self.unit_vector.x + diff_y * self.unit_vector.y)
    
    def dist_from_end_to_projection_of(self, point):
        diff_x = self.end.x - point.x
        diff_y = self.end.y - point.y
        return abs(diff_x * self.unit_vector.x + diff_y * self.unit_vector.y)
        
    def almost_equals(self, other):
        return (self.start.almost_equals(other.start) and self.end.almost_equals(other.end)) #or \
            #(self.end.almost_equals(other.start) and self.start.almost_equals(other.end))
            
    def __eq__(self, other):
        return other != None and (self.start == other.start and self.end == other.end) #or \
            #(self.end == other.start and self.start == other.end)
            
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __str__(self):
        return "start: " + str(self.start) + ". end: " + str(self.end)
                
    