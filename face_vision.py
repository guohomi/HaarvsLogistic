import cv2
import numpy as np

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('Caltech_WebFaces/pic09329.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces_result = face_detector.detectMultiScale(gray, 1.3, 5)
image_np = np.array(img)
for (x,y,w,h) in faces_result:
    cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('Face Detection', image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()