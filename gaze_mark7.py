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

    def draw_bounds(self, frame):
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

    def frame_singalar(self, frame, gray):
        eye_region = np.array(
            [
                (
                    self.landmarks.part(self.points[0]).x,
                    self.landmarks.part(self.points[0]).y,
                ),
                (
                    self.landmarks.part(self.points[2]).x,
                    self.landmarks.part(self.points[2]).y,
                ),
                (
                    self.landmarks.part(self.points[3]).x,
                    self.landmarks.part(self.points[3]).y,
                ),
                (
                    self.landmarks.part(self.points[1]).x,
                    self.landmarks.part(self.points[1]).y,
                ),
                (
                    self.landmarks.part(self.points[5]).x,
                    self.landmarks.part(self.points[5]).y,
                ),
                (
                    self.landmarks.part(self.points[4]).x,
                    self.landmarks.part(self.points[4]).y,
                ),
            ],
            np.int32,
        )

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 2)
        cv2.fillPoly(mask, [eye_region], 255)
        gray_eye = cv2.bitwise_and(gray, gray, mask=mask)
        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])

        gray_eye = gray_eye[min_y:max_y, min_x:max_x]
        #_, threshold_eye = cv2.threshold(gray_eye, 170, 255, cv2.THRESH_BINARY)
        gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        _, threshold = cv2.threshold(gray_eye, 3, 255, cv2.THRESH_BINARY_INV)
        thresh = self.thresh_processing(threshold)
        # height, width = threshold_eye.shape
        # left_side_threshold = threshold_eye[0:height, 0 : int(width / 2) : width]
        # left_side_white = cv2.countNonZero(left_side_threshold)

        # right_side_threshold = threshold_eye[0:height, int(width / 2) : width]
        # right_side_white = cv2.countNonZero(right_side_threshold)
        # eye_center = self.cal_center(
        #     thresh=thresh, mid=2, frame=gray_eye, side=self.side, draw=False
        # )

        eye_center = self.cal_center(
            threshold=thresh, mid=2, frame=gray_eye, side=self.side, draw=True
        )
        # cv2.imshow(self.side + "  threshold", threshold_eye)
        cv2.imshow(self.side + " gray", gray_eye)
        return eye_center

    # def cal_center(self, thresh, mid, frame, side, draw=False):
    #     cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     # print(cnts)
    #     cnt = max(cnts, key=cv2.contourArea)
    #     M = cv2.moments(cnt)
    #     cx = int(M["m10"] / M["m00"])
    #     cy = int(M["m01"] / M["m00"])
    #     if side == "right":
    #         cx += mid
    #     if draw:
    #         cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 5)

    #     # print(cx , cy)
    #     return [cx, cy]
    def cal_center(self, threshold, mid, frame, side, draw=False):
        # cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # # print(cnts)
        # cnt = max(cnts, key=cv2.contourArea)
        # M = cv2.moments(cnt)
        # cx = int(M["m10"] / M["m00"])
        # cy = int(M["m01"] / M["m00"])
        # if side == "right":
        #     cx += mid
        # if draw:
        #     cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 5)

        # # print(cx , cy)
        rows, cols = frame.shape
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)

            # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(frame, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 1)
            cv2.line(frame, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 1)
        return 0

    def cal_bounds(self):
        eye_height = self.center_bottom[1] - self.center_top[1]
        eye_width = self.right_point[0] - self.left_point[0]

        # print(eye_width, eye_height)
        return [eye_width, eye_height]

    def thresh_processing(self, thresh):
        thresh = cv2.erode(thresh, None, iterations=8)
        thresh = cv2.dilate(thresh, None, iterations=5)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)

        return thresh

    def mid_point(self, p1, p2):
        return [int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)]


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
                    left_eye_center = left_eye.frame_singalar(self.frame, gray)
                    #right_eye_center = right_eye.frame_singalar(self.frame, gray)

                    left_eye.draw_bounds(self.frame)
                    #right_eye.draw_bounds(self.frame)

                    mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)

                    mask = self.cal_mask(shape, mask, left_eye.points, right_eye.points)

                    eyes = cv2.bitwise_and(self.frame, self.frame, mask=mask)

                    mask = (eyes == [0, 0, 0]).all(axis=2)
                    eyes[mask] = [255, 255, 255]

                    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

                    _, thresh = cv2.threshold(eyes_gray, 180, 255, cv2.THRESH_BINARY)
                    thresh = self.thresh_processing(thresh)

                    left_eye_center = self.cal_center(
                        thresh[:, 0:mid], mid, self.frame, left_eye.side, draw=True
                    )
                    # right_eye_center = self.cal_center(
                    #     thresh[:, mid:], mid, self.frame, right_eye.side, draw=True
                    #)

                    # abs_scalar = self.cal_scalar(
                    #     left_eye.cal_bounds(), right_eye.cal_bounds()
                    # )
                    # abs_center = self.cal_abs_center(
                    #     left_eye_center, right_eye_center, abs_scalar
                    # )

                    # x3 = np.interp(
                    #     abs_center[0], (0, self.width_cam), (0, self.width_screen)
                    # )
                    # y3 = np.interp(
                    #     abs_center[1], (0, self.height_cam), (0, self.height_screen)
                    # )

                    # autopy.mouse.move(x3, y3)

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
        return [cx, cy]

    def cal_scalar(self, eye1_bounds, eye2_bounds):
        x_scalar = eye1_bounds[0] / eye2_bounds[0]
        y_scalar = eye1_bounds[1] / eye2_bounds[1]

        return [x_scalar, y_scalar]  # scale eye 2

    def cal_abs_center(self, eye1_center, eye2_center, scalar):
        # print(eye1_center)
        abs_x = (eye1_center[0] + (eye2_center[0] * scalar[0])) / 2
        abs_y = (eye1_center[1] + (eye2_center[1] * scalar[1])) / 2

        return [abs_x, abs_y]

    def close(self, cap):
        cap.release()
        cv2.destroyAllWindows()


program = Main(detector, predictor)

program.main()
