import cv2
import numpy as np
import locate
import skimage.feature as feature


# 获取图像的 glcm 纹理特征
def get_glcm(img):
    img = locate.replace_bg(img)
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    # gray = cv2.equalizeHist(gray)

    # Param:
    # source image
    # List of pixel pair distance offsets - here 1 in each direction
    # List of pixel pair angles in radians
    # graycom = feature.greycomatrix(input, [2, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)
    graycom = feature.greycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, normed=True)

    # Find the GLCM properties
    contrast = feature.greycoprops(graycom, 'contrast')
    dissimilarity = feature.greycoprops(graycom, 'dissimilarity')
    homogeneity = feature.greycoprops(graycom, 'homogeneity')
    energy = feature.greycoprops(graycom, 'energy')
    correlation = feature.greycoprops(graycom, 'correlation')
    ASM = feature.greycoprops(graycom, 'ASM')
    res = np.array([contrast, dissimilarity, homogeneity, energy, correlation, ASM])
    # print("Contrast: {}".format(contrast))
    # print("Dissimilarity: {}".format(dissimilarity))
    # print("Homogeneity: {}".format(homogeneity))
    # print("Energy: {}".format(energy))
    # print("Correlation: {}".format(correlation))
    # print("ASM: {}".format(ASM))
    return res


if __name__ == '__main__':
    img = cv2.imread(r"F:\DataSet\testImages\bazidilaohu1#\dakai-11.jpg")
    res = get_glcm(img)
    print(res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()