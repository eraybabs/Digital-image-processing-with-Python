import numpy as np

import cv2

yol = "haarcascade_frontalface_default.xml"

yuz = cv2.CascadeClassifier(yol)

biyik = cv2.imread('fotograf/biyik.png')

def biyik_ekle(biyik, fc, x, y, e, b):

    yuz_genisligi = e

    yuz_boyu = b

    biyik_genisligi = int(yuz_genisligi * 0.4166666) + 1

    biyik_eni = int(yuz_boyu * 0.142857) + 1

    biyik = cv2.resize(biyik, (biyik_genisligi, biyik_eni))

    for i in range(int(0.62857142857 * yuz_boyu), int(0.62857142857 * yuz_boyu) + biyik_eni):

        for j in range(int(0.29166666666 * yuz_genisligi), int(0.29166666666 * yuz_genisligi) + biyik_genisligi):

            for k in range(3):

                if biyik[i - int(0.62857142857 * yuz_boyu)][j - int(0.29166666666 * yuz_genisligi)][k] < 235:

                    fc[y + i][x + j][k] = \
                    biyik[i - int(0.62857142857 * yuz_boyu)][j - int(0.29166666666 * yuz_genisligi)][k]

    return fc

foto_yolu = 'fotograf/7.jpg'

kaynak = cv2.imread(foto_yolu)

gri = cv2.cvtColor(kaynak, cv2.COLOR_BGR2GRAY)

faces = yuz.detectMultiScale(

    gri,

    scaleFactor = 1.1,

    minNeighbors = 5,

    minSize = (30, 30)

)

for (x, y, e, b) in faces:

    cv2.rectangle(kaynak, (x, y), (x + e, y + b), (0, 255, 0), 2)

    kaynak = biyik_ekle(biyik, kaynak, x, y, e, b)

cv2.imshow('Biyik!', kaynak)

cv2.waitKey(0)

cv2.destroyAllWindows()