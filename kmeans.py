import os
import numpy as np
import cv2 as cv
import locate


def get_main_colors(img):
    # 替换图像背景
    image = locate.replace_bg(img)
    if image is None:
        return None
    h, w, ch = image.shape
    # 构建图像数据
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # 图像聚类
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    num_clusters = 4
    ret, label, center = cv.kmeans(data, num_clusters, None, criteria, num_clusters, cv.KMEANS_RANDOM_CENTERS)
    center = center.astype(int)

    # 将主色的bgr值转为hsv值
    hsv_colors = []
    for i in range(num_clusters):
        color = np.uint8([[center[i]]])
        hsv_color = cv.cvtColor(color, cv.COLOR_BGR2HSV)[0][0]
        hsv_colors.append(hsv_color)

    # 统计每一类的数目
    clusters = np.zeros([num_clusters], dtype=np.int32)
    for i in range(len(label)):
        clusters[label[i]] += 1
    # 计算比重
    clusters = np.float32(clusters) / float(h * w)

    # print("比重", clusters)
    # # hsv_colors = sorted(hsv_colors, key=lambda x:x[2])
    # print(hsv_colors[0], hsv_colors[1], hsv_colors[2], hsv_colors[3])
    # # 显示聚类后的图像（BGR格式）
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # dst = res.reshape(image.shape)
    # cv.imshow("img", dst)
    # # 生成主色彩条形卡片
    # card = np.zeros((50, w, 3), dtype=np.uint8)
    # # 绘制主色卡
    # center = np.int32(center)
    # x_offset = 0
    # for i in range(num_clusters):
    #     dx = np.int32(clusters[i] * w)
    #     b = int(center[i][0])
    #     g = int(center[i][1])
    #     r = int(center[i][2])
    #     cv.rectangle(card, (x_offset, 0), (x_offset + dx, 50), (b, g, r), -1)
    #     # print(r, ' ', g, ' ', b)
    #     x_offset += dx
    # cv.imshow("color", card)

    return hsv_colors, clusters
