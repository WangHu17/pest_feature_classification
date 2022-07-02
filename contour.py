import os

import cv2
import locate


# 轮廓逼近
def get_contour_approx(img):
    # 获取图像轮廓
    contour = locate.getContour(img)
    # img = cv2.resize(img, (600, 600))
    # 轮廓长度
    arcLen = cv2.arcLength(contour, True)
    # 轮廓逼近
    points = cv2.approxPolyDP(contour, 0.03*arcLen, True)
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


if __name__ == '__main__':
    # path = "F:\\DataSet\\baseImgs\\"
    # out = "F:\\DataSet\\contourApprox1\\"
    # for i in os.listdir(path):
    #     img = cv2.imread(os.path.join(path, i))
    #     img = contour_approx(img)
    #     cv2.imwrite(os.path.join(out, i), img)
    basePath = "F:\\DataSet\\contourBase"
    targetPath = "F:\\DataSet\\baseImgs"
    bases = get_base_contours(basePath)
    index = 0
    for i in os.listdir(targetPath):
        filepath = os.path.join(targetPath, i)
        img = cv2.imread(filepath)
        res = match_contour(bases, img)
        newName = targetPath + "\\" + res + str(index) + ".jpg"
        index = index + 1
        os.rename(filepath, newName)
    # path = "F:\\DataSet\\contourBase"
    # for i in os.listdir(path):
    #     img = cv2.imread(os.path.join(path, i))
    #     c, res = get_contour_approx(img)
    #     cv2.imshow(i, res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # targetPath = "F:\\DataSet\\baseImgs\\1 (1).jpg"
    # target = cv2.imread(targetPath)
    # match_contour(bases, target)