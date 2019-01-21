import numpy as np
import cv2
import signal
import matplotlib.pyplot as plt
from skimage.transform import resize

cap = cv2.VideoCapture(0)

class Detect(object):
    
    def __init__(self,
                 threshold = True,
                 hand_name = 'hand1.xml',
                 face_name = 'haarcascade.xml'):
        
        self.min_length = 100
        self.max_length = 500
        
        self.threshold = True
        
        self.margin = 10
        self.imgs = []
        self.X = []
        self.Y = []
        
        self.hand_cascade = cv2.CascadeClassifier(hand_name)
        self.face_cascade = cv2.CascadeClassifier(face_name)
        
    def detect_hand(self, frame):
        
        output_list = []
        
        blur = cv2.GaussianBlur(frame, (5,5) ,0.1) 
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) 
        
        if self.threshold:
            val2,thresh1 = cv2.threshold(gray,170,255,cv2.THRESH_BINARY) 
            hand = self.hand_cascade.detectMultiScale(thresh1, 1.3, 5)
        else:
            hand = self.hand_cascade.detectMultiScale(gray, 1.3, 5)
        #mask = np.zeros(thresh1.shape, dtype = "uint8") 
        
        if type(hand) is None:
            return None
    
        for x, y, w, h in hand:
            if w > self.min_length and w < self.max_length \
                and h > self.min_length and h < self.max_length:
                
                self.X.append(x)
                self.Y.append(y)
                
                output_list.append([x,y,w,h])
                
        if len(output_list) > 0:
            return output_list
        
        return None
    
    def detect_face(self, frame):
        
        faces = self.face_cascade.detectMultiScale(frame,
                                     scaleFactor=1.1,
                                     minNeighbors=3,
                                     minSize=(100, 100))
        
        if len(faces) != 0:
            face = faces[0]
            (x, y, w, h) = face
            left = x - self.margin // 2
            right = x + w + self.margin // 2
            bottom = y - self.margin // 2
            top = y + h + self.margin // 2
            
            img = frame[bottom:top, left:right, :]
            img = resize(img,
                         (160, 160), mode='reflect')
            
            self.imgs.append(img)
            
            shape = [(left - 1, bottom - 1), (right + 1, top + 1)]
            
            return img
        
        return None
            
    
hand_cascade = cv2.CascadeClassifier('hand1.xml')

det = Detect()

while True:
    
    try:
        ret, frame = cap.read()
        
        cv2.imshow('image', frame)
        
        out_hand = det.detect_hand(frame)
        out_face = det.detect_face(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if out_hand is None or out_face is None:
            continue

        #second_frame = np.copy(frame)
        for index, hand in enumerate(out_hand):
            
            x, y, w, h = hand
            cv2.imshow('frame_hand_{}'.format(index), frame[y:y+h, x:x+w])
        
        cv2.imshow('frame_face', out_face)

    except Exception as e: 
        print(e)
        break

cap.release()
cv2.destroyAllWindows()
