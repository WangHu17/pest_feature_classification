import os
import cv2
import numpy as np
import sobel

# 设置最小框的面积
THRESHHOLD = 30


# 通过聚类获取黑色背景的害虫二值图
def get_thresh_by_kmeans(img):
    img = cv2.medianBlur(img, 11)
    # cv2.imwrite(r"F:\imwrite_imgs\img.jpg", img)
    # 构建图像数据
    data = img.reshape((-1, 3))
    data = np.float32(data)
    # 图像聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    num_clusters = 2
    ret, label, center = cv2.kmeans(data, num_clusters, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    # 生成聚类后的图像
    center = np.uint8(center)
    res = center[label.flatten()]
    dst = res.reshape(img.shape)
    # cv2.imshow("dst", dst)
    # cv2.imwrite(r"F:\imwrite_imgs\dst.jpg", dst)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.imwrite(r"F:\imwrite_imgs\gray.jpg", gray)
    # 获取灰度均值
    threshold = int(np.mean(gray)) + 1
    th, thres = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thres", thres)
    # cv2.imwrite(r"F:\imwrite_imgs\thres.jpg", thres)
    return thres


# 通过聚类获取害虫轮廓
def get_contour1(img):
    image = get_thresh_by_kmeans(img)
    h, w = image.shape
    row1 = int(h - h * 0.02)
    col1 = int(w * 0.02)
    row2 = int(h - h * 0.02)
    col2 = int(w - w * 0.02)
    v1 = image[row1, col1]
    v2 = image[row2, col2]
    if v1 == 255 and v2 == 255:
        cv2.bitwise_not(image, image)
    thresh = cv2.GaussianBlur(image, (5, 5), 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    # cv2.imshow("contour", contour)
    # cv2.imwrite(r"F:\imwrite_imgs\contour.jpg", contour)
    # 获取轮廓索引
    max_area = 0
    maxI = 0
    if len(contours) == 0:
        print("获取轮廓失败")
        return None
    for index in range(len(contours)):
        area = cv2.contourArea(contours[index])
        if area > max_area:
            max_area = area
            maxI = index
    return contours[maxI]


# 获取害虫轮廓
def get_contour(img):
    h = img.shape[0]
    w = img.shape[1]
    # print(h, w)
    row1 = int(h - h * 0.02)
    col1 = int(w * 0.02)
    row2 = int(h - h * 0.02)
    col2 = int(w - w * 0.02)
    val1 = int(np.mean(img[1, 1]))
    val2 = int(np.mean(img[h-1, 1]))
    if val1 < 40 and val2 < 40:
        thresh = get_thresh_by_kmeans(img)
    else:
        # thresh为二值化之后的黑白图片。
        thresh = sobel.sobel_cal(img, THRESHHOLD)
        # cv2.imshow("image", image)
        v1 = thresh[row1, col1]
        v2 = thresh[row2, col2]
        # print(v1, v2)
        if v1 == 255 and v2 == 255:
            thresh = get_thresh_by_kmeans(img)
    v1 = thresh[row1, col1]
    v2 = thresh[row2, col2]
    if v1 == 255 and v2 == 255:
        cv2.bitwise_not(thresh, thresh)

    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
    # cv2.imshow("thresh ", thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    # cv2.imshow("contour", contour)
    # cv2.imwrite(r"F:\imwrite_imgs\baieContour.jpg", contour)

    # 获取轮廓索引
    max_area = 0
    maxI = 0
    if len(contours) == 0:
        print('获取轮廓失败')
        return None
    for index in range(len(contours)):
        area = cv2.contourArea(contours[index])
        if area > max_area:
            max_area = area
            maxI = index
    return contours[maxI]


# 获取害虫子图像
def get_pest_img(img):
    # 获取图像轮廓
    contour = get_contour(img)
    if contour is None:
        return None
    # 截取图片
    x, y, w, h = cv2.boundingRect(contour)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    img = img[y:y + h, x:x + w]
    return img


# 返回替换背景后的害虫图像
def replace_bg(img):
    # 获取图像轮廓
    contour = get_contour(img)
    if contour is None:
        return None
    # 替换轮廓外的背景色
    # fill_color = [120, 200, 70]
    fill_color = [0, 255, 0]
    mask_value = 255
    mask = np.zeros(img.shape[:-1]).astype(np.uint8)
    cv2.fillPoly(mask, [contour], mask_value)
    sel = mask != mask_value
    img[sel] = fill_color

    # cv2.imshow("replace_bg", img)
    # cv2.imwrite(r"F:\imwrite_imgs\replace_bg.jpg", img)
    # 截取图片
    x, y, w, h = cv2.boundingRect(contour)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.imshow("rectangle", img)
    # cv2.imwrite(r"F:\imwrite_imgs\rectangle.jpg", img)
    img = img[y:y + h, x:x + w]
    # cv2.imshow("cut", img)
    # cv2.imwrite(r"F:\imwrite_imgs\cut.jpg", img)

    return img


if __name__ == '__main__':
    path = r'F:\DataSet\testImages\ganlanyee1#\1 (120).jpg'
    path1 = r'D:\PestLib\weifenleilinchimu\00060A0B.jpg'
    path2 = r'F:\DataSet\testImages\bazidilaohu1#\sanjiao1-41.jpg'
    # img = cv2.imread(path)
    # img = cv2.resize(img, (400, 400))
    # img = get_pest_img(img)
    # for i in os.listdir(path):
    #     img = cv2.imread(os.path.join(path, i))
    #     img = cv2.resize(img, (400, 400))
    #     contour = get_contour(img)
    #     contourImg = cv2.drawContours(img, contour, -1, (0, 255, 0), 2)
    #     cv2.imshow(i, contourImg)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow(i)
        # if img is None:
        #     print(i)
        #     continue
        # contourImg = cv2.drawContours(img, contour, -1, (0, 255, 0), 2)
        # cv2.imshow(i, img)
    img = cv2.imread(path2)
    img = cv2.resize(img, (200, 200))
    # out = get_contour(img)
    out = replace_bg(img)
    cv2.imshow("img", out)
    # cv2.imwrite(r"F:\imwrite_imgs\out1.jpg", out)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
