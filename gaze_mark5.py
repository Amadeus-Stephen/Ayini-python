#!/usr/bin/env python3
import cv2
import dlib
import math
from math import hypot
from math import pi, pow
import numpy as np

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class Eye_base:
    def __init__(self, side,points ):
        self.side   = side
        self.points = points

    def blink(self, frame):
        hor_line_length = hypot((self.left_point[0] - self.right_point[0]),   (self.left_point[1] - self.right_point[1])) 
        ver_line_length = hypot((self.center_top[0] - self.center_bottom[0]), (self.center_top[1] - self.center_bottom[1]))

        ratio = hor_line_length / ver_line_length

        return ratio
class Main:
    def __init__(self, detector , predictor):
        self.detector  = detector
        self.predictor = predictor
