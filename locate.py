import cv2
import numpy as np

from sobel import sobel_cal

# 设置最小框的面积
THRESHHOLD = 30


def run(img):
    img = cv2.resize(img, (600, 600))
    # img为裁剪过后的图片,image为二值化之后的黑白图片。
    img, image = sobel_cal(img, THRESHHOLD)

    thresh = cv2.GaussianBlur(image, (5, 5), 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 替换轮廓外的背景色
    fill_color = [0, 255, 0]
    mask_value = 255
    mask = np.zeros(img.shape[:-1]).astype(np.uint8)
    cv2.fillPoly(mask, contours, mask_value)
    sel = mask != mask_value
    img[sel] = fill_color

    # 获取害虫轮廓索引
    max = 0
    maxI = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max:
            max = area
            maxI = i

    # 截取图片
    x, y, w, h = cv2.boundingRect(contours[maxI])
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    img = img[y:y+h, x:x+w]

    # 展示框标点结果
    # cv2.imshow("thresh ", thresh)
    # cv2.imshow("replace", img)

    return img