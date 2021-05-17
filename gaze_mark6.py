#!/usr/bin/env python3
import cv2
import dlib
import math
from math import hypot
from math import pi, pow
import numpy as np
import autopy


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class Eye_base:
    def __init__(self, side, points, thresh, landmarks):
        self.side = side
        self.points = points
        self.thresh = thresh

        self.landmarks = landmarks

        self.left_point = (
            self.landmarks.part(self.points[0]).x,
            self.landmarks.part(self.points[0]).y,
        )
        self.right_point = (
            self.landmarks.part(self.points[1]).x,
            self.landmarks.part(self.points[1]).y,
        )
        self.center_top = self.mid_point(
            self.landmarks.part(self.points[2]), self.landmarks.part(self.points[3])
        )
        self.center_bottom = self.mid_point(
            self.landmarks.part(self.points[4]), self.landmarks.part(self.points[5])
        )

    def blink(self, frame):
        hor_line_length = hypot(
            (self.left_point[0] - self.right_point[0]),
            (self.left_point[1] - self.right_point[1]),
        )
        ver_line_length = hypot(
            (self.center_top[0] - self.center_bottom[0]),
            (self.center_top[1] - self.center_bottom[1]),
        )

        ratio = hor_line_length / ver_line_length

        return ratio

    def draw_lines(self, frame):
        # hor_line = cv2.line(frame , self.left_point, self.right_point   , (255,255,255),2)
        # ver_line = cv2.line(frame , self.center_top, self.center_bottom , (255,255,255),2)

        # print(self.left_point)
        # print(self.center_bottom)
        # print(self.right_point)
        # print(self.center_top)
        cv2.rectangle(
            frame,
            (self.left_point[0], self.center_top[1]),
            (self.right_point[0], self.center_bottom[1]),
            (255, 0, 0),
        )

    def mid_point(self, p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


class Main:
    def __init__(self, detector, predictor):
        self.detector = detector
        self.predictor = predictor
        self.cap = cv2.VideoCapture(-1)
        self.width_cam = 640
        self.height_cam = 480
        self.width_screen = autopy.screen.size()[0]
        self.height_screen = autopy.screen.size()[1]

    def main(self):

        if self.cap.isOpened():
            while True:
                _, self.frame = self.cap.read()
                self.frame = cv2.flip(self.frame, 1)

                thresh = self.frame.copy()
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray)

                for rect in rects:
                    shape = self.shape_to_np(predictor(gray, rect))

                    mid = (shape[42][0] + shape[39][0]) // 2

                    landmarks = predictor(gray, rect)

                    left_eye = Eye_base(
                        "left", [36, 39, 37, 38, 41, 40], thresh[:, 0:mid], landmarks
                    )
                    right_eye = Eye_base(
                        "right", [42, 45, 43, 44, 47, 46], thresh[:, mid:], landmarks
                    )

                    left_eye.draw_lines(self.frame)
                    right_eye.draw_lines(self.frame)
                    mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)

                    mask = self.cal_mask(shape, mask, left_eye.points, right_eye.points)

                    eyes = cv2.bitwise_and(self.frame, self.frame, mask=mask)

                    mask = (eyes == [0, 0, 0]).all(axis=2)
                    eyes[mask] = [255, 255, 255]

                    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

                    _, thresh = cv2.threshold(eyes_gray, 170, 255, cv2.THRESH_BINARY)
                    thresh = self.thresh_processing(thresh)

                    left_eye_center = self.cal_center(
                        thresh[:, 0:mid], mid, self.frame, left_eye.side, draw=True
                    )
                    right_eye_center = self.cal_center(
                        thresh[:, mid:], mid, self.frame, right_eye.side, draw=True
                    )

                cv2.imshow("eyes", self.frame)
                # cv2.imshow("thresh", thresh)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.cap.release()
                    break
            self.close(self.cap)

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def cal_mask(self, shape, mask, left, right):
        mask = self.cal_eye_on_mask(shape, mask, left)
        mask = self.cal_eye_on_mask(shape, mask, right)

        return mask

    def cal_eye_on_mask(self, shape, mask, side):
        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)

        return mask

    def thresh_processing(self, thresh):
        thresh = cv2.erode(thresh, None, iterations=8)
        thresh = cv2.dilate(thresh, None, iterations=5)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)

        return thresh

    def cal_center(self, thresh, mid, frame, side, draw=False):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(cnts)
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if side == "right":
            cx += mid
        if draw:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), 2)

        # print(cx , cy)
        return (cx, cy)

    def cal_scalar(self, eye1_ray, r2, eye2_ray, direct):
        eye1_distance = eye1_ray[0][direct] - eye1_ray[1][direct]
        eye2_distance = eye2_ray[0][direct] - eye2_ray[1][direct]

        return eye1_distance / eye2_distance  # scale eye 2

    def cal_abs_center():
        print()
        # cal and return mid point of both eyes here

    def close(self, cap):
        cap.release()
        cv2.destroyAllWindows()


program = Main(detector, predictor)

program.main()
