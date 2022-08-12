import cv2
import numpy as np


def fill_hole(im_in, threshhold):
    # Threshold.
    # Set values equal to or above 40 to 255.
    # Set values below 40 to 0.
    th, im_th = cv2.threshold(im_in, threshhold, 255, cv2.THRESH_BINARY)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_out


def baweraopen(image, size):
    '''
    @image:单通道二值图，数据类型uint8
    @size:欲去除区域大小(黑底上的白区域)
    '''
    output = image.copy()
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    for i in range(1, nlabels - 1):
        regions_size = stats[i, 4]
        if regions_size < size:
            x0 = stats[i, 0]
            y0 = stats[i, 1]
            x1 = stats[i, 0] + stats[i, 2]
            y1 = stats[i, 1] + stats[i, 3]
            for row in range(y0, y1):
                for col in range(x0, x1):
                    if labels[row, col] == i:
                        output[row, col] = 0
    return output


def sobel_cal(imgs, threshhold):
    # 高斯模糊去噪
    img = cv2.medianBlur(imgs, 7)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    color_space = S
    # Sobel算子
    x = cv2.Sobel(color_space, cv2.CV_16S, 1, 0)  # 对x求一阶导
    y = cv2.Sobel(color_space, cv2.CV_16S, 0, 1)  # 对y求一阶导
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # b.设置卷积核5*5
    kernel = np.ones((2, 2), np.uint8)

    # 腐蚀的作用说白了就是让暗的区域变大，而膨胀的作用就是让亮的区域变大
    # 图像的膨胀
    dst = cv2.dilate(sobel, kernel)

    # 空洞填充
    out = fill_hole(dst, threshhold)

    # c.图像的腐蚀，默认迭代次数
    erosion = cv2.erode(out, kernel)

    # 图像的膨胀
    dst = cv2.dilate(erosion, kernel)

    # 去除小的斑点
    dst = baweraopen(dst, 300)

    return dst
