import cv2
import dlib 
import numpy as np 
from math import hypot
import imutils



cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)





while True:
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(40), landmarks.part(41))
        
        #hor_line = cv2.line(frame, left_point, right_point, (255,0 , 0), 2)
        #ver_line = cv2.line(frame, center_top, center_bottom, (255, 0, 0), 2)


        hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))   #calcul de la distance de la ligne horizentale
        ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        
        x = (left_point[0]) + int(hor_line_lenght / 2) 
        y = (center_bottom[1] - int(ver_line_lenght / 2))
        #cv2.circle(frame,(x,y), 2,(0,0,255),2)


        roi = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        min_x = np.min(roi[:, 0]) 
        max_x = np.max(roi[:, 0])
        min_y = np.min(roi[:, 1])
        max_y = np.max(roi[:, 1])
        eye = frame[min_y:max_y, min_x:max_x]
        gray_eye = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.GaussianBlur(gray_eye,(7,7),40)
        
        circles = cv2.HoughCircles(gray_eye,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=10,minRadius=30,maxRadius=50)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:

                center = (i[0], i[1])
            # circle center
                cv2.circle(eye, center, 1, (0, 100, 100), 3)
            # circle outline
                radius = i[2]
                cv2.circle(eye, center, radius, (0, 0, 255), 3)

        #roi_only = cv2.resize(roi_only,None, fx=7, fy=7)
        cv2.imshow("roi", eye)
        
    cv2.imshow("Frame", frame)
    cv2.imshow("gray", gray)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()