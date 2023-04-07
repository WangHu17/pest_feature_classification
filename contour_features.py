import os
import cv2
import locate
import math
import numpy as np
import sobel


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
    # cv2.drawContours(img, [points], -1, (0, 255, 0), 2)
    # cv2.imshow("approx", img)
    # cv2.imwrite(r"F:\imwrite_imgs\baieContour1.jpg", img)
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


def get_contour_features(img):
    # 获取图像轮廓
    contour = locate.get_contour(img)

    features = []

    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])  # X方向的重心
    cy = int(M['m01']/M['m00'])  # Y方向的重心
    features.append(cx)
    features.append(cy)

    perimeter = cv2.arcLength(contour, True)  # 周长
    features.append(perimeter)
    # print("周长：", perimeter)

    area = cv2.contourArea(contour)  # 面积
    features.append(area)
    # print("面积：", area)

    area = cv2.contourArea(contour)  # 轮廓面积
    hull = cv2.convexHull(contour)  # 计算出凸包形状
    hull_area = cv2.contourArea(hull)  # 计算凸包面积
    solidity = float(area) / hull_area  # 轮廓面积与凸包面积的比
    features.append(solidity)
    # print("轮廓面积与凸包面积的比：", solidity)

    contour_approx = get_contour_approx(img)
    hull1 = cv2.convexHull(contour_approx, returnPoints=False)
    defects = cv2.convexityDefects(contour_approx, hull1)
    defects_num = 0 if defects is None else len(defects)
    features.append(defects_num)
    # print("凸性缺陷：", defects_num)

    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)  # MA,ma 分别为长轴短轴长度
    features.append(MA)
    features.append(ma)
    # print("长轴短轴长度：", MA, ma)
    # print("-----------------------------------")
    return features


# 获取轮廓特征
def get_contour_features1(img):
    # 获取图像轮廓
    contour = locate.get_contour(img)

    features = []

    area = cv2.contourArea(contour)  # 面积
    features.append(area)
    # print("面积：", area)

    perimeter = cv2.arcLength(contour, True)  # 周长
    features.append(perimeter)
    # print("周长：", perimeter)

    (x, y), (w, l), angle = cv2.fitEllipse(contour)
    features.append(l)  # 长轴
    features.append(w)  # 短轴
    # print("长轴短轴长度：", l, w)

    sp = perimeter * perimeter / (4 * area * math.pi)  # 形状参数
    features.append(sp)
    # print("形状参数：", sp)

    r = 4 * area / (math.pi * l * l)  # 似圆度
    features.append(r)
    # print("似圆度：", r)

    ec = np.sqrt(1.0 - (w / l) ** 2)  # 偏心率
    features.append(ec)
    # print("偏心率：", ec)

    d = area / (l * w)  # 占空比
    features.append(d)
    # print("占空比：", d)

    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])  # X方向的重心
    cy = int(M['m01'] / M['m00'])  # Y方向的重心
    features.append(cx)
    features.append(cy)
    # print("重心：", cx, cy)

    area = cv2.contourArea(contour)  # 轮廓面积
    hull = cv2.convexHull(contour)  # 计算出凸包形状
    # cv2.polylines(img, [hull], True, (0, 255, 0), 2)
    hull_area = cv2.contourArea(hull)  # 计算凸包面积
    solidity = float(area) / hull_area  # 轮廓面积与凸包面积的比
    features.append(solidity)
    # print("轮廓面积与凸包面积的比：", solidity)

    contour = get_contour_approx(img)
    hull1 = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull1)
    defects_num = 0 if defects is None else len(defects)
    features.append(defects_num)
    # print("凸性缺陷：", defects_num)

    # humoments = hu(img)
    # for humoment in humoments:
    #     features.append(humoment)

    # print(area, perimeter, l, w, sp, r, ec, d, (cx, cy), solidity, defects_num)

    return features


# Hu不变矩
def hu(img):
    thresh = sobel.sobel_cal(img, 30)
    moments = cv2.moments(thresh)
    humoments = cv2.HuMoments(moments)
    humoments = -(np.log(np.abs(humoments))) / np.log(10)  # 取对数
    return humoments


if __name__ == '__main__':
    img = cv2.imread(r"F:\DataSet\svm_train_imgs\15\001416C0.jpg")
    path = r"F:\imwrite_imgs\img"
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i))
        img = cv2.resize(img, (200, 200))
        hu(img)
    # img = cv2.resize(img, (400, 400))
    # hu(img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
