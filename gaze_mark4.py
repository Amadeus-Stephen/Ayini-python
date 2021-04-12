import cv2
import dlib
import numpy as np



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
kernel = np.ones((9,9), np.uint8)



class Eye:
    def __init__(self,side ,points):
        self.points = points
        self.name = side + " eye"

    def get_points(self):

        return self.points


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68,2), dtype=dtype)
    for i in range(0 , 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def render_mask(mask, left , right):
    mask = eye_on_mask(eye_on_mask(mask, left), right)
    mask = cv2.dilate(mask, kernel, 5)

    return mask


def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)

    return mask


def thresh_processing(thresh):
    thresh = cv2.erode(thresh, None, iterations=8) #1
    thresh = cv2.dilate(thresh, None, iterations=5) #2
    thresh = cv2.medianBlur(thresh, 3) #3
    thresh = cv2.bitwise_not(thresh)

    return thresh



def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass

cap = cv2.VideoCapture(0)
while True:
        _, img = cap.read()
        thresh = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray)

        for rect in rects:
            left_eye  = Eye("left" , [36, 37, 38, 39, 40, 41])
            right_eye = Eye("right", [42, 43, 44, 45, 46, 47])
            shape = shape_to_np(predictor(gray, rect))
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

            mask = render_mask(mask , left_eye.get_points() , right_eye.get_points())

            eyes = cv2.bitwise_and(img, img, mask=mask)

            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

            _, thresh = cv2.threshold(eyes_gray, 170, 255, cv2.THRESH_BINARY)
            thresh = thresh_processing(thresh)

            mid = (shape[42][0] + shape[39][0]) // 2
            contouring(thresh[:, 0:mid], mid, img)
            #contouring(thresh[:, mid:], mid, img, True)

        # for (x,y) in shape[36:48]:
        #     cv2.circle(img, (x,y), 2, (255, 0 , 0), -1)
        cv2.imshow("thresh", thresh)
        cv2.imshow("eyes", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
