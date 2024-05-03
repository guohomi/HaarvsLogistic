import numpy as np
import cv2
from os import listdir
import matplotlib.pyplot as plt

face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for f in listdir("Caltech_WebFaces"):
    img = cv2.imread('Caltech_WebFaces/' + f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_result = face_detector.detectMultiScale(gray, 1.3, 5)
    print('File name:' + f + '. Face count: ' + str(len(faces_result)))


cv2.destroyAllWindows()