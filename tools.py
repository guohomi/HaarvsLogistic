import cv2
import random

def imread(filename):
    return cv2.imread(filename, 0)

def normalize(t):
    return (t - t.mean()) / t.std()


def random_split(dataset, training_proportion):
    random.shuffle(dataset)
    return (
        dataset[:int(training_proportion * len(dataset))],
        dataset[int(training_proportion * len(dataset)):])

def hist_256(t):
    hist = [0] * 256
    for v in t:
        hist[int(v)] += 1
    return hist

