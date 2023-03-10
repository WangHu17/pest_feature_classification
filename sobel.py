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


def sobel_cal(img, threshhold):
    # 中值滤波去噪
    img = cv2.medianBlur(img, 7)
    # 高斯滤波去噪
    # img = cv2.GaussianBlur(img, (3, 3), 1)
    # 均值滤波去噪
    # img = cv2.blur(img, (5, 5))
    # cv2.imshow("1-medianBlur", img)
    # cv2.imwrite(r"F:\imwrite_imgs\1-Blur.jpg", img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow("2-hsv", hsv)
    # cv2.imwrite(r"F:\imwrite_imgs\2-hsv.jpg", hsv)
    H, S, V = cv2.split(hsv)
    color_space = S

    # Sobel算子
    x = cv2.Sobel(color_space, cv2.CV_16S, 1, 0)  # 对x求一阶导
    y = cv2.Sobel(color_space, cv2.CV_16S, 0, 1)  # 对y求一阶导
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # laplacian算子
    # img_temp = cv2.Laplacian(color_space, cv2.CV_16S)
    # img_Laplacian = cv2.convertScaleAbs(img_temp)

    # Canny边缘检测
    # threshold1 = 0
    # threshold2 = 160
    # img_Canny = cv2.Canny(img, threshold1, threshold2)

    # cv2.imshow("3-sobel", img_Laplacian)
    # cv2.imwrite(r"F:\imwrite_imgs\1.jpg", img_Canny)

    # b.设置卷积核5*5
    kernel = np.ones((2, 2), np.uint8)

    # 腐蚀的作用说白了就是让暗的区域变大，而膨胀的作用就是让亮的区域变大
    # 图像的膨胀
    dst = cv2.dilate(sobel, kernel)
    # cv2.imshow("dilate", dst)

    # 空洞填充
    dst = fill_hole(dst, threshhold)
    # cv2.imshow("4-fill_hole", dst)
    # cv2.imwrite(r"F:\imwrite_imgs\2.jpg", dst)

    # c.图像的腐蚀，默认迭代次数
    erosion = cv2.erode(dst, kernel)
    #
    # 图像的膨胀
    dst = cv2.dilate(erosion, kernel)

    # 去除小的斑点
    dst = baweraopen(dst, 300)

    return dst


if __name__ == '__main__':
    path1 = r'F:\DataSet\testImages\ganlanyee1#\1 (39).jpg'
    # path1 = r'F:\DataSet\testImages\ganlanyee1#\1 (23).jpg'
    path2 = r'F:\DataSet\testImages\ganlanyee1#\1 (116).jpg'
    # path3 = r'F:\DataSet\testImages\ganlanyee1#\1 (51).jpg'
    path3 = r'F:\DataSet\testImages\ganlanyee1#\1 (120).jpg'
    path4 = r'D:\PestLib\weifenleilinchimu\000609E0.jpg'
    img = cv2.imread(path4)
    img = cv2.resize(img, (200, 200))
    thresh = sobel_cal(img, 30)
    cv2.imwrite(r"F:\imwrite_imgs\thresh1.jpg", thresh)
    # cv2.imshow("sobel_cal", thresh)
    cv2.waitKey(0)