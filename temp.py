import cv2
import numpy as np
import locate
from matplotlib import pyplot as plt
fig = plt.figure()


# 直方图均衡化
def equalizeHist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.subplot(221), plt.imshow(img, 'gray'), plt.title('gray1')
    plt.subplot(222), plt.hist(img.ravel(), 256, [0, 256]),
    plt.title('Histogram'), plt.xlim([0, 256])
    equ = cv2.equalizeHist(img)
    hist1 = cv2.calcHist([equ], [0], None, [256], [0, 256])
    plt.subplot(223), plt.imshow(equ, 'gray'), plt.title('gray2')
    plt.subplot(224), plt.hist(equ.ravel(), 256, [0, 256]),
    plt.title('EqualizeHist'), plt.xlim([0, 256])
    fig.tight_layout(pad=0.4, w_pad=0, h_pad=0)
    # plt.savefig(r"F:\imwrite_imgs\EqualizeHist.jpg")
    plt.show()


def depart(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, thres = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("thres", thres)
    # cv2.imwrite(r"F:\imwrite_imgs\hist4.jpg", thres)


# 直方图平均化
def histAverage(img1, img2):
    hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
    plt.subplot(321), plt.imshow(img1, 'gray'), plt.title('gray1')
    plt.subplot(322), plt.hist(img1.ravel(), 256, [0, 256]),
    plt.title('Histogram1'), plt.xlim([0, 256])

    hist1 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    plt.subplot(323), plt.imshow(img2, 'gray'), plt.title('gray2')
    plt.subplot(324), plt.hist(img2.ravel(), 256, [0, 256]),
    plt.title('Histogram2'), plt.xlim([0, 256])

    histAve = []
    for i in range(0, len(hist)):
        histAve.append((hist[i] + hist1[i]) / 2)
    plt.subplot(326), plt.plot(histAve),
    plt.title('HistAverage'), plt.xlim([0, 256])
    fig.tight_layout(pad=0.4, w_pad=0, h_pad=0)
    plt.savefig(r"F:\imwrite_imgs\HistAverage1.jpg")
    plt.show()


if __name__ == '__main__':
    path = r"F:\DataSet\testImages\bazidilaohu1#\hhs-bazidilaohu3-24.jpg"
    path1 = r"F:\DataSet\testImages\bazidilaohu2#\hei-bazidilaohu0-18.jpg"
    img = cv2.imread(path)
    img1 = cv2.imread(path1)
    img = cv2.resize(img, (400, 400))
    img1 = cv2.resize(img1, (400, 400))
    img = locate.get_pest_img(img)
    img1 = locate.get_pest_img(img1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(r"F:\imwrite_imgs\equalizeHist.jpg", img)
    histAverage(img, img1)
    # depart(img, 130)
    cv2.waitKey(0)
