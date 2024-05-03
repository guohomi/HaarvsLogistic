import cv2
from os import listdir
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
tpr = []
fpr = []
t_count = 0
f_count = 0
total_count = 0
tpr.append(0)
fpr.append(0)
y_true = []
y_score = []
for f in listdir("Caltech_WebFaces"):
    img = cv2.imread('Caltech_WebFaces/' + f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_result = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(faces_result) > 0:
        t_count = t_count + 1
        y_score.append(1)
    else:
        f_count = f_count + 1
        y_score.append(0)
    total_count = total_count + 1
    y_true.append(1)
    if total_count%100 == 0:
        tpr.append(t_count/total_count)
        #fpr.append(f_count/f_count)
        fpr.append(0 / 7093)
    print('File name:' + f + '. Face count: ' + str(len(faces_result)))
for f in listdir("negative"):
    img = cv2.imread('negative/' + f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_result = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(faces_result) > 0:
        t_count = t_count + 1
        y_score.append(1)
    else:
        f_count = f_count + 1
        y_score.append(0)
    total_count = total_count + 1
    y_true.append(0)
    if total_count % 100 == 0:
        tpr.append(t_count / total_count)
        # fpr.append(f_count/f_count)
        fpr.append(0 / 7093)
    print('File name:' + f + '. Face count: ' + str(len(faces_result)))
#create ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

cv2.destroyAllWindows()