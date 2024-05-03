from os import listdir
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from trainlogisticmodel import *

def extract(filename, clf):
    image = cv2.imread(filename, 0)
    feat = pre_processing(image)
    lst = []
    if clf.predict([feat])[0] == 1:
        lst.append((clf.predict_proba([feat])[0, 1]))
    return lst

if __name__ == '__main__':
    t_count = 0
    f_count = 0
    total_count = 0
    y_true = []
    y_score = []
    #load the model
    f = open("logistic_model.mdl", "rb")
    clf = pickle.load(f)
    f.close()
    print('Processing positives:')
    positives=listdir("Caltech_WebFaces")
    for f in tqdm(positives):
        result = extract('Caltech_WebFaces/' + f, clf)
        if len(result) > 0:
            t_count = t_count + 1
            y_score.append(1)
        else:
            f_count = f_count + 1
            y_score.append(0)
        total_count = total_count + 1
        y_true.append(1)
    print('Processing negatives:')
    negatives = listdir("negative")
    for f in negatives:
        result = extract('negative/' + f, clf)
        if len(result) > 0:
            t_count = t_count + 1
            y_score.append(1)
        else:
            f_count = f_count + 1
            y_score.append(0)
        total_count = total_count + 1
        y_true.append(0)

    print('Creating ROC curve...')
    # create ROC curve
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


