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

        self.landmarks = landmarks #imports landmarks point from machine learning model

        self.left_point = ( # defines the points on which the the left point of the eye is located
            self.landmarks.part(self.points[0]).x,
            self.landmarks.part(self.points[0]).y,
        )
        self.right_point = ( # defines the points on which the the right point of the eye is located
            self.landmarks.part(self.points[1]).x,
            self.landmarks.part(self.points[1]).y,
        )
        self.center_top = self.mid_point( # since the top eye in the model consists of 2 points,
                                          # we need to find the mid point of the points
            self.landmarks.part(self.points[2]), self.landmarks.part(self.points[3])
        )
        self.center_bottom = self.mid_point(# since the bottom eye in the model consists of 2 points,
                                          # we need to find the mid point of the points
            self.landmarks.part(self.points[4]), self.landmarks.part(self.points[5])
        )

    def blink(self, frame): #finds the distance of the top and bottom of the eye to detect if the eye is closed

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

    def draw_lines(self, frame): # draws rectangles around the location of the eyes
        # hor_line = cv2.line(frame , self.left_point, self.right_point   , (255,255,255),2)
        # ver_line = cv2.line(frame , self.center_top, self.center_bottom , (255,255,255),2)
         
        cv2.rectangle(
            frame,
            (self.left_point[0], self.center_top[1]),
            (self.right_point[0], self.center_bottom[1]),
            (255, 0, 0),
        )

    def frame_singular(self, frame, gray):

        # eye region is the location of the eyes in a specific frame given
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

        height, width, _ = frame.shape                      # all of this is isolating the eye area of the eye region given
        mask = np.zeros((height, width), np.uint8)          # the points of the eye.
        cv2.polylines(mask, [eye_region], True, 255, 2)     # 
        cv2.fillPoly(mask, [eye_region], 255)               #
        gray_eye = cv2.bitwise_and(gray, gray, mask=mask)   #
        min_x = np.min(eye_region[:, 0])                    #
        max_x = np.max(eye_region[:, 0])                    #
        min_y = np.min(eye_region[:, 1])                    #
        max_y = np.max(eye_region[:, 1])                    #

        gray_eye = gray_eye[min_y:max_y, min_x:max_x]       #

        # gray_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)

        # cv2.imshow(self.side + " eye", gray_eye)
        gray_eye = cv2.medianBlur(gray_eye, 9)
        _, threshold = cv2.threshold(gray_eye, 3, 255, cv2.THRESH_BINARY_INV)   #returns a version of the gray scaled eye
                                                                                #into a binary black and white
        thresh = self.cal_thresh(threshold)

        eye_center = self.cal_center(
            threshold=thresh, mid=2, frame=gray_eye, side=self.side, draw=True
        )
        rows = gray_eye.shape[0]
        circles = cv2.HoughCircles(gray_eye,cv2.HOUGH_GRADIENT,1,15,param1=100,param2=10, minRadius=20 , maxRadius=50)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                
                cv2.circle(gray_eye, center, 1, (255, 0, 0), 3)
            # circle outline
                # radius = i[2]
                # cv2.circle(gray_eye, center, radius, (0, 0, 255), 3)
        # cv2.imshow(self.side + "  threshold", threshold_eye)
        img_not = cv2.bitwise_not(gray_eye)
       
        image = np.zeros((self.cal_bounds()[1], self.cal_bounds()[0], 3), np.uint8)
        image[:] = (255, 255, 255)
        #cv2.imshow(self.side + " red", image)
        
        gray_eye = cv2.resize(gray_eye, self.cal_bounds())

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # returns gray scale
        print("image")
        print(len(image.shape))
        print("eye")
        print(len(gray_eye.shape))
        # dst = cv2.addWeighted(image,0.7,gray_eye,0.3,0)
        cv2.imshow('dst',image)
        #`cv2.imshow(self.side + " img_not", img_not)



        return eye_center

    def cal_bounds(self): # calculates the bonds of the eye for the scaling that is need later
        self.left_point = (
            self.landmarks.part(self.points[0]).x,
            self.landmarks.part(self.points[0]).y,
        )
        self.right_point = (
            self.landmarks.part(self.points[1]).x,
            self.landmarks.part(self.points[1]).y,
        )
        self.center_top =self.mid_point(
            self.landmarks.part(self.points[2]), self.landmarks.part(self.points[3])
        )
        self.center_bottom = self.mid_point(
            self.landmarks.part(self.points[4]), self.landmarks.part(self.points[5])
        )
        eye_height = self.center_bottom[1] - self.center_top[1]
        eye_width = self.right_point[0] - self.left_point[0]

        #print(eye_width, eye_height)
        return [eye_width, eye_height]

    def cal_thresh(self, thresh): # returns the black and white binary after "cleaning"
        thresh = cv2.erode(thresh, None, iterations=8)
        thresh = cv2.dilate(thresh, None, iterations=5)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)

        return thresh

    def cal_center(self, threshold, mid, frame, side, draw=False): #calculates the center of the eye based 
                                                                   #on the black and white subset
        rows, cols = frame.shape
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)

            hor_line = [(0, y + int(h / 2)), (cols, y + int(h / 2))]
            # cv2.line(frame, hor_line[0], hor_line[1] , (0, 255, 0), 1)

            ver_line = [(x + int(w / 2), 0), (x + int(w / 2), rows)]
            # cv2.line(frame,  ver_line[0] , ver_line[1], (0, 255, 0), 1)

            hor_line_midpoint = self.mid_point(hor_line[0], hor_line[1], idx_cal=True) 
            ver_line_midpoint = self.mid_point(ver_line[0], ver_line[1], idx_cal=True)

            true_center = (hor_line_midpoint[0],ver_line_midpoint[1]) # I dont think I though of this earlier
                                                                #but this would return to points as ((x,y),(x,y))
                                                                # where we need (x,y)

            # cv2.circle(frame,(true_center),2,(255,0,0),3)
        return true_center

    def mid_point(self, p1, p2, idx_cal=False):
        if idx_cal:
            return [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]
        else:
            return [int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)]



class Main:
    def __init__(self, detector, predictor):
        self.detector = detector
        self.predictor = predictor
        self.cap = cv2.VideoCapture(0) #video input from camera
        self.width_cam = 640
        self.height_cam = 480
        self.width_screen = autopy.screen.size()[0]
        self.height_screen = autopy.screen.size()[1]
        self.frame_hist = [0,0,0,0,0,0,0,0]# a center point history of the past 5 frames

    def main(self):

        while True:
            ret, self.frame = self.cap.read()
            self.frame = cv2.flip(self.frame, 1)
            if ret is False:
                break

            roi = self.frame
            thresh = self.frame.copy()
            rows, cols, _ = roi.shape

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # returns gray scale
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0) # cleans image of fractions
            
            _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
            rects = detector(gray_roi)
            for rect in rects: # for all faces do function
                shape = self.shape_to_np(predictor(gray_roi, rect)) # returns the gray scaled face into a 2d array 
                mid = (shape[42][0] + shape[39][0]) // 2 # mid of face
                landmarks = predictor(gray_roi, rect) # array

                left_points  = [36, 39, 37, 38, 41, 40] # all points around the left and right eye look at the 68dat.png
                right_points = [42, 45, 43, 44, 47, 46]
                
                left_eye = Eye_base("left", left_points, thresh[:, 0:mid], landmarks)
                right_eye = Eye_base("right", right_points, thresh[:, 0:mid], landmarks)

                #finds center of the eyes
                left_eye_center = left_eye.frame_singular(self.frame, gray_roi)
                right_eye_center = right_eye.frame_singular(self.frame, gray_roi)

                # draws retangles around each eye
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

                # returns difference of eyes for a focused point
                abs_scalar = self.cal_scalar(
                    left_eye.cal_bounds(), right_eye.cal_bounds()
                )

                # returns the abs center of the eye to the camera
                abs_center = self.cal_abs_center(
                    left_eye_center , right_eye_center , abs_scalar
                )

                # ratio the size of the camera to the res of the monitor/display from the multi the abs_center coords
                abs_ratio = self.cal_ratio_scalar(left_eye.cal_bounds(),right_eye.cal_bounds())


                abs_center_scaled = [
                    abs_center[0] * abs_ratio[0][0] * abs_ratio[1][0],
                    abs_center[1] * abs_ratio[0][1] * abs_ratio[1][1]
                ]

                # create a history index and render the average of the past 5 frames
                #print("bounds", left_eye.cal_bounds() , right_eye.cal_bounds())

                #print("scalar", abs_scalar)
                #print("center", abs_center)

                #print("center_scaled", abs_center_scaled)

                # avg_hist = self.cal_avg_hist(abs_center_scaled)

                # if avg_hist:
                #     autopy.mouse.move(avg_hist[0], avg_hist[1])
                # else:
                #     autopy.mouse.move(abs_center_scaled[0], abs_center_scaled[1])


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


    def cal_scalar(self, eye1_bounds, eye2_bounds):                                             # finds the difference in Width and Height of the eyes
        x_scalar = eye1_bounds[0] / eye2_bounds[0]                                              # To be then used to math into a singlar point in cal_abs_center()
        y_scalar = eye1_bounds[1] / eye2_bounds[1]

        return [x_scalar, y_scalar]  # scale eye 2


    def cal_abs_center(self, eye1_center, eye2_center, scalar):                                 # finds the midpoint of x and y with the scaled 
        #print(eye1_center)                                                                      # difference factor of the other eye to equalize the number
        abs_x = (eye1_center[0] + (eye2_center[0] * scalar[0])) / 2
        abs_y = (eye1_center[1] + (eye2_center[1] * scalar[1])) / 2

        return [abs_x, abs_y]

    def cal_ratio_scalar(self, eye1_bounds , eye2_bounds):                                     #finds the comparitive bounds of both eyes to make and ABS bound
                                                                                               #The scales to camera then montior
        abs_bounds = [(eye1_bounds[0] + eye2_bounds[0])/2 , (eye1_bounds[1] + eye2_bounds[1])/2 ]

        ratio_scalar_cam = [ self.width_cam / abs_bounds[0] , self.height_cam / abs_bounds[1] ]
        ratio_scalar_mon = [ self.width_screen / self.width_cam , self.height_screen / self.height_cam]

        return [ratio_scalar_cam , ratio_scalar_mon]

    def cal_avg_hist(self, n_coords):
        del self.frame_hist[0]
        self.frame_hist.append(n_coords)
        #self.frame_hist[len(self.frame_hist) -1 ] = n_coords

        #sum(self.frame_hist) / len(self.frame_hist) not how coords work lol
        if 0 not in self.frame_hist:

            list_x = []
            list_y = []
            for coords in self.frame_hist:
                list_x.append(coords[0])
                list_y.append(coords[1])

            avg_x = sum(list_x) / len(self.frame_hist)
            avg_y = sum(list_y) / len(self.frame_hist)

            return [avg_x, avg_y]
        else:
            return 0
            
    def close(self, cap):
        cap.release()
        cv2.destroyAllWindows()


program = Main(detector, predictor)

program.main()
