import pickle

from tools import *
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
import os
import numpy as np

NEGATIVES = "negative/"
POSITIVES = "Caltech_WebFaces/"

def pre_processing(img):
    img = cv2.resize(img, (64, 64))
    return feature_extraction(img)

def feature_extraction(img):
    lbp_pattern = local_binary_pattern(img, 8, 3)
    blocks = []
    for i in range(8):
        for j in range(8):
            blocks.append(lbp_pattern[i*8:(i+1)*8, j*8:(j+1)*8])
    blocks = map(lambda M: M.reshape((1, M.size))[0], blocks)
    blocks = map(lambda M: normalize(np.array(hist_256(M))), blocks)
    out = np.array([], dtype=float)
    for v in blocks:
        out = np.concatenate((out, v))
    return out

if __name__ == '__main__':
    #loading dataset
    positives = []
    negatives = []
    positives_names = os.listdir(POSITIVES)
    negatives_names = os.listdir(NEGATIVES)

    print("load positives...")

    for img_name in tqdm(positives_names):
        try:
            positives.append((imread(POSITIVES + img_name), 1))
        except Exception as e:
            print(e)
            pass

    print("load negatives...")

    for img_name in tqdm(negatives_names):
        try:
            negatives.append((imread(NEGATIVES + img_name), 0))
        except Exception as e:
            print(e)
            pass

    dataset = positives + negatives

    #slitting dataset
    print("sliting dataset...")
    train_set, test_set = random_split(dataset, .8)

    X_test,  y_test  = zip(*test_set)
    X_train, y_train = zip(*train_set)

    #feature extraction (LBP histogram of the image)
    print("feature extraction... (LBP block histogram)")
    tmp = []
    for v in X_test:
        try:
            tmp.append(pre_processing(v))
        except Exception as e:
            print(e)
            pass
    X_test = tmp

    tmp = []
    for v in X_train:
        try:
            tmp.append(pre_processing(v))
        except:
            pass
    X_train = tmp

    print(len(X_test))
    print(len(X_train))

    #creating a model (Logistic classifier)
    print("model training...")
    clf = LogisticRegression()
    clf.fit(np.array(X_train), np.array(y_train))
    print("accuracy : {}".format(
        sum(clf.predict(np.array(X_test)) == y_test) / float(len(y_test))))

    #save model
    print("model saving...")
    f = open("logistic_model.mdl", "wb")
    pickle.dump(clf, f)
    f.close()


