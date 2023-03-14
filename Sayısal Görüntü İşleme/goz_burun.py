import math

import skimage.exposure

import cv2

import numpy as np

img = cv2.imread("fotograf/7.jpg")

x = 130

y = 109

yaricap = 30

artis = 1.5

kirp = img[y-yaricap:y+yaricap, x-yaricap:x+yaricap]

boy, en = kirp.shape[:2]

xcent = en / 2

ycent = boy / 2

rad = min(xcent,ycent)

map_x = np.zeros((boy, en), np.float32)

map_y = np.zeros((boy, en), np.float32)

maske = np.zeros((boy, en), np.uint8)

for y in range(boy):

    Y = (y - ycent)/ycent

    for x in range(en):

        X = (x - xcent)/xcent

        R = math.hypot(X,Y)

        if R == 0:

            map_x[y, x] = x

            map_y[y, x] = y

            maske[y,x] = 255

        elif R >= .90:

            map_x[y, x] = x

            map_y[y, x] = y

            maske[y,x] = 0

        elif artis >= 0:

            map_x[y, x] = xcent * X * math.pow((2 / math.pi) * (math.asin(R)/R), artis) + xcent

            map_y[y, x] = ycent * Y * math.pow((2 / math.pi) * (math.asin(R)/R), artis) + ycent

            maske[y, x] = 255

        elif artis < 0:

            artis2 = -artis

            map_x[y, x] = xcent * X * math.pow((math.sin(math.pi * R / 2) / R), artis) + xcent

            map_y[y, x] = ycent * Y * math.pow((math.sin(math.pi * R / 2) / R), artis2) + ycent

            maske[y,x] = 255

carp = cv2.remap(kirp, map_x, map_y, cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

bulanik = 7

maske = cv2.copyMakeBorder(maske, bulanik, bulanik, bulanik, bulanik, borderType=cv2.BORDER_CONSTANT, value=(0))

maske = cv2.GaussianBlur(maske, (0, 0), sigmaX = bulanik, sigmaY = bulanik, borderType = cv2.BORDER_DEFAULT)

b, e = maske.shape

maske = maske[bulanik: b-bulanik, bulanik: e-bulanik]

maske = cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR)

maske = skimage.exposure.rescale_intensity(maske, in_range=(127.5, 255), out_range=(0, 1))

carpili = (carp * maske + kirp * (1 - maske)).clip(0, 255).astype(np.uint8)

result = img.copy()

result[y-yaricap: y+yaricap, x-yaricap: x+yaricap] = carpili

cv2.imwrite("fotograf/filtreli7.jpg", result)

cv2.imshow('img', img)

cv2.imshow('kirp', kirp)

cv2.imshow('carp', carp)

cv2.imshow('maske', maske)

cv2.imshow('carpili', carpili)

cv2.imshow('result', result)

cv2.waitKey(0)

cv2.destroyAllWindows()