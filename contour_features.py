import os
import cv2
import locate


# 轮廓逼近
def get_contour_approx(img):
    # 获取图像轮廓
    contour = locate.get_contour(img)
    # img = cv2.resize(img, (600, 600))
    # 轮廓长度
    arcLen = cv2.arcLength(contour, True)
    # 轮廓逼近
    points = cv2.approxPolyDP(contour, 0.01*arcLen, True)
    # print(points)
    # 绘制轮廓逼近
    # cv2.drawContours(img, [points], -1, (0, 255, 0), 2)
    # cv2.imshow("轮廓逼近", img)
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


if __name__ == '__main__':
    path = r"F:\DataSet\svm_training_imgs\0"
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i))
        img = cv2.resize(img, (600, 600))
        features = get_contour_features(img)
        print(features)
        # cv2.imshow(i, contourImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
