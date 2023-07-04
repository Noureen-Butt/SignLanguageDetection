import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 30
imgSize =400
counter = 0

folder = "Data/Yes "

classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ['Hello', 'Iloveyou', 'Yes', 'No']

while True:
    success, img = capture.read()
    ImageOut = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h= hand['bbox']

        imageWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imageCrope = img[y-offset:y + h+offset, x-offset:x + w+offset]

        # imageWhite[0:imageCrope.shape[0], 0:imageCrope.shape[1]] = imageCrope
        # imageWhite[0:imageCrope.shape[0],0:imageCrope.shape[1]] = imageCrope
        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imageResize = cv2.resize(imageCrope, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imageWhite[:, wGap:wCal + wGap] = imageResize
            prediction, index = classifier.getPrediction(imageWhite)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imageResize = cv2.resize(imageCrope, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imageWhite[hGap:hCal + hGap, :] = imageResize
            prediction, index = classifier.getPrediction(imageWhite)

        cv2.putText(ImageOut, labels[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(ImageOut, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

        cv2.imshow("CropppedImage", imageCrope)
        cv2.imshow("WhiteImage", imageWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
