import cv2
import numpy as np
import sobel

# 设置最小框的面积
THRESHHOLD = 30


# 通过聚类获取黑色背景的害虫二值图
def handle_black_background_pic(img):
    img = cv2.medianBlur(img, 11)
    # cv2.imshow("模糊", img)
    # 构建图像数据
    data = img.reshape((-1, 3))
    data = np.float32(data)
    # 图像聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    num_clusters = 2
    ret, label, center = cv2.kmeans(data, num_clusters, None, criteria, num_clusters, cv2.KMEANS_RANDOM_CENTERS)
    center = center.astype(int)
    # 显示聚类后的图像
    center = np.uint8(center)
    res = center[label.flatten()]
    dst = res.reshape(img.shape)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    th, thres = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)
    # thres = sobel.baweraopen(thres, 300)
    return thres


# 获取害虫轮廓
def getContour(img):
    img = cv2.resize(img, (600, 600))
    # img为缩放过后的原图,image为二值化之后的黑白图片。
    img, image = sobel.sobel_cal(img, THRESHHOLD)
    v1 = image[590, 10]
    v2 = image[590, 590]
    if v1 == 255 and v2 == 255:
        image = handle_black_background_pic(img)
        v1 = image[590, 10]
        v2 = image[590, 590]
        if v1 == 255 and v2 == 255:
            cv2.bitwise_not(image, image)

    thresh = cv2.GaussianBlur(image, (5, 5), 0)
    # cv2.imshow("thresh ", thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # 获取轮廓索引
    max = 0
    maxI = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max:
            max = area
            maxI = i
    return contours[maxI]


# 返回替换背景后的害虫图像
def replaceBG(img):
    # 获取图像轮廓
    contour = getContour(img)
    img = cv2.resize(img, (600, 600))

    # 替换轮廓外的背景色
    fill_color = [0, 255, 0]
    mask_value = 255
    mask = np.zeros(img.shape[:-1]).astype(np.uint8)
    cv2.fillPoly(mask, [contour], mask_value)
    sel = mask != mask_value
    img[sel] = fill_color

    # 截取图片
    x, y, w, h = cv2.boundingRect(contour)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    img = img[y:y+h, x:x+w]

    return img


if __name__ == '__main__':
    path = "F:\\DataSet\\baseImgs\\1 (43).jpg"
    img = cv2.imread(path)
    out = replaceBG(img)
    cv2.imshow("img", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()