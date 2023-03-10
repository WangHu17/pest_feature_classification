import os
import cv2
import locate
import math
import numpy as np


# 轮廓逼近
def get_contour_approx(img):
    # 获取图像轮廓
    contour = locate.get_contour(img)
    # img = cv2.resize(img, (600, 600))
    # 轮廓长度
    arcLen = cv2.arcLength(contour, True)
    # 轮廓逼近
    points = cv2.approxPolyDP(contour, 0.01 * arcLen, True)
    # print(points)
    # 绘制轮廓逼近
    # cv2.drawContours(img, [points], -1, (0, 0, 255), 2)
    # cv2.imshow("approx", img)
    return points


# 获取基准图像的逼近轮廓
def get_base_contours(path):
    bases = {}
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i))
        contour = get_contour_approx(img)
        name = i.split('.')[0]
        bases[name] = contour
    return bases


# 比较轮廓
def match_contour(bases, target):
    targetContour = get_contour_approx(target)
    maxVal = 100
    res = ""
    for name in bases.keys():
        val = cv2.matchShapes(bases.get(name), targetContour, 1, 0)
        if maxVal > val:
            maxVal = val
            res = name
    return res


# 获取轮廓特征
def get_contour_features(img):
    # 获取图像轮廓
    contour = locate.get_contour(img)

    features = []

    area = cv2.contourArea(contour)  # 面积
    features.append(area)
    print("面积：", area)

    perimeter = cv2.arcLength(contour, True)  # 周长
    features.append(perimeter)
    print("周长：", perimeter)

    (x, y), (w, l), angle = cv2.fitEllipse(contour)
    features.append(l)  # 长轴
    features.append(w)  # 短轴
    print("长轴短轴长度：", l, w)

    sp = perimeter * perimeter / (4 * area * math.pi)  # 形状参数
    print("形状参数：", sp)

    r = 4 * area / (math.pi * l * l)  # 似圆度
    print("似圆度：", r)

    ec = np.sqrt(1.0 - (w / l) ** 2)  # 偏心率
    print("偏心率：", ec)

    d = area / (l * w)  # 占空比
    print("占空比：", d)

    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])  # X方向的重心
    cy = int(M['m01'] / M['m00'])  # Y方向的重心
    features.append(cx)
    features.append(cy)
    print("重心：", cx, cy)

    area = cv2.contourArea(contour)  # 轮廓面积
    hull = cv2.convexHull(contour)  # 计算出凸包形状
    # cv2.polylines(img, [hull], True, (0, 255, 0), 2)
    hull_area = cv2.contourArea(hull)  # 计算凸包面积
    solidity = float(area) / hull_area  # 轮廓面积与凸包面积的比
    features.append(solidity)
    print("轮廓面积与凸包面积的比：", solidity)

    # cv2.imshow("img", img)

    contour = get_contour_approx(img)
    hull1 = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull1)
    defects_num = 0 if defects is None else len(defects)
    features.append(defects_num)
    print("凸性缺陷：", defects_num)

    return features


if __name__ == '__main__':
    # path = r"F:\DataSet\svm_training_imgs\0"
    # for i in os.listdir(path):
    #     img = cv2.imread(os.path.join(path, i))
    #     img = cv2.resize(img, (600, 600))
    #     features = get_contour_features(img)
    #     print(features)
    # cv2.imshow(i, contourImg)
    path = r'D:\PestLib\meiguobaie\baie1.1\00141711.jpg'
    path1 = r'F:\DataSet\testImages\ganlanyee1#\1 (120).jpg'
    img = cv2.imread(path1)
    img = cv2.resize(img, (200, 200))
    get_contour_features(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
