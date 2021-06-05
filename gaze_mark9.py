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
        # self.center_bottom = self.mid_point(
        #     self.landmarks.part(self.points[4]), self.landmarks.part(self.points[5])
        # )

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

    def frame_singalar(self, frame, gray):

        # RE WRITE THIS BITCH TO WHERE IT FOLLOWS THE NOTES SQUARE
        
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
        #eye_region = np.array(
        #     [
        #         (
        #             self.landmarks.part(self.points[0]).x,
        #             self.landmarks.part(self.points[1]).y,
        #         ),
        #         (
        #             self.landmarks.part(self.points[3]).x,
        #             self.landmarks.part(self.points[3]).y,
        #         ),
        #         (
        #             self.landmarks.part(self.points[2]).x,
        #             self.landmarks.part(self.points[1]).y,
        #         ),
        #         (
        #             self.landmarks.part(self.points[0]).x,
        #             self.landmarks.part(self.points[1]).y,
        #         ),
        #         (
        #             self.landmarks.part(self.points[0]).x,
        #             self.landmarks.part(self.points[3]).y,
        #         ),
        #         (
        #             self.landmarks.part(self.points[3]).x,
        #             self.landmarks.part(self.points[3]).y,
        #         ),
        #     ],
        #     # np.int32,
        # )
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
        # gray_eye = gray_eye
        # _, threshold_eye = cv2.threshold(gray_eye, 170, 255, cv2.THRESH_BINARY)
        gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        _, threshold = cv2.threshold(gray_eye, 3, 255, cv2.THRESH_BINARY_INV)
        thresh = self.cal_thresh(threshold)
        # height, width = threshold_eye.shape
        # left_side_threshold = threshold_eye[0:height, 0 : int(width / 2) : width]
        # left_side_white = cv2.countNonZero(left_side_threshold)

        # right_side_threshold = threshold_eye[0:height, int(width / 2) : width]
        # right_side_white = cv2.countNonZero(right_side_threshold)
        eye_center = self.cal_center(
            threshold=thresh, mid=2, frame=gray_eye, side=self.side, draw=True
        )

        # cv2.imshow(self.side + "  threshold", threshold_eye)
        cv2.imshow(self.side + " gray", gray_eye)
        return eye_center

    def cal_bounds(self):
        eye_height = self.center_bottom[1] - self.center_top[1]
        eye_width = self.right_point[0] - self.left_point[0]

        print(eye_width, eye_height)
        return [eye_width, eye_height]

    def cal_thresh(self, thresh):
        thresh = cv2.erode(thresh, None, iterations=8)
        thresh = cv2.dilate(thresh, None, iterations=5)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)

        return thresh

    def cal_center(self, threshold, mid, frame, side, draw=False):
        rows, cols = frame.shape
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)

            hor_line = [(0, y + int(h / 2)), (cols, y + int(h / 2))]
            cv2.line(frame, hor_line[0], hor_line[1] , (0, 255, 0), 1)

            ver_line = [(x + int(w / 2), 0), (x + int(w / 2), rows)]
            cv2.line(frame,  ver_line[0] , ver_line[1], (0, 255, 0), 1)

            hor_line_midpoint = self.mid_point(hor_line[0], hor_line[1], idx_cal=True)
            ver_line_midpoint = self.mid_point(ver_line[0], ver_line[1], idx_cal=True)

        return [hor_line_midpoint, ver_line_midpoint]

    def mid_point(self, p1, p2, idx_cal=False):
        if idx_cal:
            return [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]
        else:
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

        while True:
            ret, self.frame = self.cap.read()
            self.frame = cv2.flip(self.frame, 1)
            if ret is False:
                break

            roi = self.frame
            thresh = self.frame.copy()
            rows, cols, _ = roi.shape

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
            _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
            rects = detector(gray_roi)
            for rect in rects:
                shape = self.shape_to_np(predictor(gray_roi, rect))
                mid = (shape[42][0] + shape[39][0]) // 2
                landmarks = predictor(gray_roi, rect)

                left_points  = [36, 39, 37, 38, 41, 40]
                right_points = [42, 45, 43, 44, 47, 46]
                
                left_eye = Eye_base("left", left_points, thresh[:, 0:mid], landmarks)
                right_eye = Eye_base("right", right_points, thresh[:, 0:mid], landmarks)


                left_eye_center = left_eye.frame_singalar(self.frame, gray_roi)
                right_eye_center = right_eye.frame_singalar(self.frame, gray_roi)
                cv2.rectangle(
                    roi,
                    (landmarks.part(left_points[0]).x, landmarks.part(left_points[1]).y),
                    (landmarks.part(left_points[3]).x, landmarks.part(left_points[3]).y),
                    (255, 0, 0),
                )
                cv2.rectangle(
                    roi,
                    (landmarks.part(right_points[0]).x, landmarks.part(right_points[1]).y),
                    (landmarks.part(right_points[3]).x, landmarks.part(right_points[3]).y),
                    (255, 0, 0),
                )


            cv2.imshow("Roi", roi)

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

    def cal_thresh(self, thresh):
        thresh = cv2.erode(thresh, None, iterations=8)
        thresh = cv2.dilate(thresh, None, iterations=5)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)

        return thresh

    def cal_center(self, thresh, mid, frame, side, draw=False):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
        print(eye1_center)
        abs_x = (eye1_center[0] + (eye2_center[0] * scalar[0])) / 2
        abs_y = (eye1_center[1] + (eye2_center[1] * scalar[1])) / 2

        return [abs_x, abs_y]

    def close(self, cap):
        cap.release()
        cv2.destroyAllWindows()


program = Main(detector, predictor)

program.main()
