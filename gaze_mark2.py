import cv2
import numpy as np
import dlib
from math import hypot
from math import pi , pow
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class Eye:
    def __init__(self,name,landmarks,pointArray) :
        self.name          = name
        self.pointArray    = pointArray
        self.landmarks     = landmarks
        self.left_point    = (self.landmarks.part(self.pointArray[0]).x , self.landmarks.part(self.pointArray[0]).y)
        self.right_point   = (self.landmarks.part(self.pointArray[1]).x , self.landmarks.part(self.pointArray[1]).y)
        self.center_top    = midPoint(self.landmarks.part(self.pointArray[2]), self.landmarks.part(self.pointArray[3]))
        self.center_bottom = midPoint(self.landmarks.part(self.pointArray[4]), self.landmarks.part(self.pointArray[5]))
    def drawLines(self , frame):
        hor_line = cv2.line(frame , self.left_point, self.right_point   , (255,255,255),2)
        ver_line = cv2.line(frame , self.center_top, self.center_bottom , (255,255,255),2)

    def blink(self):
        hor_line_length = hypot((self.left_point[0] - self.right_point[0])  , (self.left_point[1] - self.right_point[1]))
        ver_line_length = hypot((self.center_top[0] - self.center_bottom[0]), (self.center_top[1] - self.center_bottom[1]))

        ratio = hor_line_length / ver_line_length

        # if ratio > 5.7:
        #     cv2.putText(frame,"BLINK" , (50 , 150), font , 3 , (0,0,255))
        #
        return ratio

    def get_gaze_ratio(self, frame,gray):
        left_eye_region = np.array([(self.landmarks.part(self.pointArray[0]).x, self.landmarks.part(self.pointArray[0]).y),
                                    (self.landmarks.part(self.pointArray[2]).x, self.landmarks.part(self.pointArray[2]).y),
                                    (self.landmarks.part(self.pointArray[3]).x, self.landmarks.part(self.pointArray[3]).y),
                                    (self.landmarks.part(self.pointArray[1]).x, self.landmarks.part(self.pointArray[1]).y),
                                    (self.landmarks.part(self.pointArray[5]).x, self.landmarks.part(self.pointArray[5]).y),
                                    (self.landmarks.part(self.pointArray[4]).x, self.landmarks.part(self.pointArray[4]).y)], np.int32)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 170, 255, cv2.THRESH_BINARY)
        height , width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width/2): width]
        left_side_white = cv2.countNonZero(left_side_threshold)


        right_side_threshold = threshold_eye[0: height, int(width/2) : width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        # gaze_ratio = 1 if left_side_white == 0 else (left_side_white/right_side_white)
        # gaze_ratio = 5 if right_side_white ==0 else (left_side_white/right_side_white)


        if left_side_white ==0:
            gaze_ratio = 1
        if right_side_white ==0:
            gaze_ratio =5
        else:
            gaze_ratio = (left_side_white/right_side_white)

        #cv2.imshow(self.name, gray_eye)



        #cv2.imshow("Right eye", right_side_threshold)
        #cv2.imshow("Left eye", left_side_threshold)
        cv2.imshow(self.name + "  threshold", threshold_eye)
        return gaze_ratio

def midPoint(p1, p2):
    return int ((p1.x + p2.x)/2), int ((p1.y + p2.y)/2)

def main(detector , predictor):
    print("(!)press any key to close")
    while True:
        _, frame = cap.read()
        #new_frame = np.zeros((500 , 500 , 3) , np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        font = cv2.FONT_HERSHEY_COMPLEX
        for face in faces:
            x , y = face.left() , face.top()
            x1 , y1 = face.right() , face.bottom()
            cv2.rectangle(frame , (x ,y ) , (x1 , y1),(255 ,255 ,255))
            landmarks = predictor(gray, face)

            #land marks in the order of  l ,r ,ct1,ct2 ,cb1,cb2
            right_eye  = Eye("Right_Eye",landmarks, [36,39,37,38,41,40])
            left_eye   = Eye("Left_Eye" ,landmarks, [42,45,43,44,47,46])
            left_eye_gaze_ratio  =  left_eye.get_gaze_ratio(frame,gray)
            right_eye_gaze_ratio =  right_eye.get_gaze_ratio(frame,gray)


            #true_gaze_ratio = (right_eye_gaze_ratio + left_eye_gaze_ratio) / 2


            # if true_gaze_ratio <= 1:
            #     cv2.putText(frame, "Right" , (50 , 100), font , 2, (0,0,255),3)
            #     new_frame[:] = (0 , 0,255)
            # elif 1 < true_gaze_ratio <1.7:
            #     cv2.putText(frame, "CENTER" , (50 , 100) , font, 2, (0,0,255),3)
            # else:
            #     new_frame[:] = (255, 0,0)
            #     cv2.putText(frame,"LEFT" , (50 , 100) , font , 2, (0,0,255), 3)
            #right_eye.drawLines(frame)
            #left_eye.drawLines(frame)

        cv2.imshow("Frame", frame)
        #cv2.imshow("New Frame", new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


main(detector, predictor)
