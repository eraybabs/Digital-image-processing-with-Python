import dlib

import cv2

import numpy as np

bul = dlib.get_frontal_face_detector()

tahmin = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def empty(a):

    pass

cv2.namedWindow('BGR')

cv2.resizeWindow('BGR', 640, 240)

cv2.createTrackbar('Blue', 'BGR', 0, 255, empty)

cv2.createTrackbar('Green', 'BGR', 0, 255, empty)

cv2.createTrackbar('Red', 'BGR', 0, 255, empty)

def createBox(img, points, scale = 5, masked = False, cropped = True):

    if masked:

        maske = np.zeros_like(img)

        maske = cv2.fillPoly(maske, [points], (255, 255, 255))

        foto = cv2.bitwise_and(img, maske)

    if cropped:

        bbox = cv2.boundingRect(points)

        x, y, w, h = bbox

        kirp = img[y:y+h, x:x+w]

        kirp = cv2.resize(kirp, (0, 0), None, scale, scale)

        return kirp

    else:

        return maske

while True:

    foto = cv2.imread('fotograf/7.jpg')

    foto = cv2.resize(foto, (0, 0), None, 0.5, 0.5)

    fotoOrijinal = foto.copy()

    fotoGray = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)

    faces = bul(fotoGray)

    for face in faces:

        x1, y1 = face.left(), face.top()

        x2, y2 = face.right(), face.bottom()

        landmarks = tahmin(fotoGray, face)

        myPoints =[]

        for n in range(68):

            x = landmarks.part(n).x

            y = landmarks.part(n).y

            myPoints.append([x,y])

        myPoints = np.array(myPoints)

        dudak = createBox(foto, myPoints[48:61], 8, masked = True, cropped = False)

        dudak_rengi = np.zeros_like(dudak)

        b = cv2.getTrackbarPos('Blue', 'BGR')

        g = cv2.getTrackbarPos('Green', 'BGR')

        r = cv2.getTrackbarPos('Red', 'BGR')

        dudak_rengi[:] = b, g, r

        dudak_rengi = cv2.bitwise_and(dudak, dudak_rengi)

        dudak_rengi = cv2.GaussianBlur(dudak_rengi, (7,7), 10)

        dudak_rengi = cv2.addWeighted(fotoOrijinal, 1, dudak_rengi, 0.4, 0)

        cv2.imshow('BGR', dudak_rengi)

        print(myPoints)

    cv2.imshow("Original", fotoOrijinal)

    cv2.waitKey(1)