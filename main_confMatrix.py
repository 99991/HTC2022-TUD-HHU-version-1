# This file is from
# https://fips.fi/HTCdata.php
# It has been modified so that loadImg can also load 8 bit grayscale images.

import sys

import numpy as np
import skimage


def loadImg(imgFile):
    # load image and convert to grayscale array
    img = skimage.io.imread(imgFile)

    # if the image is not grayscale already, convert it to grayscale
    if len(img.shape) == 3:
        img = img[:, :, :3]  # removes 4th channel if present (alpha channel)
        img = skimage.color.rgb2gray(img)  # converts to grayscale

    # forces binary image
    threshold = 0.5
    img[img > threshold] = 1.0
    img[img <= threshold] = 0.0

    # convert to bool
    img = img.astype(bool)
    # fig = skimage.io.imshow(img)
    # plt.show()

    return img


def calcScore(reconImgFile, groundtruthImgFile):
    Ir = loadImg(reconImgFile)
    It = loadImg(groundtruthImgFile)

    AND = lambda x, y: np.logical_and(x, y)
    NOT = lambda x: np.logical_not(x)

    # confusion matrix
    TP = float(len(np.where(AND(It, Ir))[0]))
    TN = float(len(np.where(AND(NOT(It), NOT(Ir)))[0]))
    FP = float(len(np.where(AND(NOT(It), Ir))[0]))
    FN = float(len(np.where(AND(It, NOT(Ir)))[0]))
    cmat = np.array([[TP, FN], [FP, TN]])

    # Matthews correlation coefficient (MCC)
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if denominator == 0:
        score = 0
    else:
        score = numerator / denominator

    return score


if __name__ == "__main__":

    if sys.version_info.major == 2:
        sys.stdout.write("Sorry! This program requires Python 3.x\n")
        sys.exit(1)

    imgREcon = './recon_10x10.png'
    imgGroundTruth = './groundTruth_10x10.png'

    score = calcScore(imgREcon, imgGroundTruth)

    print('reconstruction score: %f' % score)
