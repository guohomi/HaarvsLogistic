from cv2 import imread, IMREAD_GRAYSCALE
from numpy import hsplit, vsplit, array
from numpy import float32, arange, repeat, newaxis

def split_images(img_name, img_size):
    # Load the full image from the specified file
    img = imread(img_name, IMREAD_GRAYSCALE)

    # Find the number of sub-images on each row and column according to their size
    num_rows = img.shape[0] / img_size
    num_cols = img.shape[1] / img_size

    # Split the full image horizontally and vertically into sub-images
    sub_imgs = [hsplit(row, num_cols) for row in vsplit(img, num_rows)]

    return img, array(sub_imgs)

def split_data(img_size, sub_imgs, ratio):
    # Compute the partition between the training and testing data
    partition = int(sub_imgs.shape[1] * ratio)

    # Split dataset into training and testing sets
    train = sub_imgs[:, :partition, :, :]
    test = sub_imgs[:, partition:sub_imgs.shape[1], :, :]

    # Flatten each image into a one-dimensional vector
    train_imgs = train.reshape(-1, img_size ** 2)
    test_imgs = test.reshape(-1, img_size ** 2)

    # Create the ground truth labels
    labels = arange(10)
    train_labels = repeat(labels, train_imgs.shape[0] / labels.shape[0])[:, newaxis]
    test_labels = repeat(labels, test_imgs.shape[0] / labels.shape[0])[:, newaxis]

    return train_imgs, train_labels, test_imgs, test_labels