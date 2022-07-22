import cv2
import numpy as np
import pylab as pl
import locate
import skimage.feature as feature
from skimage.feature import local_binary_pattern
import pandas as pd


# 构建gabor滤波器
def build_filters():
    filters = []
    # ksize = [7, 9, 11, 13, 15, 17]  # gabor尺度，6个
    ksize = [3, 5, 7, 9, 11]  # gabor尺度，5个
    lamda = np.pi / 2.0  # 波长
    theta = [0, np.pi / 2, np.pi / 4, np.pi - np.pi / 4]  # gabor方向，0°，45°，90°，135°，共四个
    for i in range(4):
        for K in range(5):
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta[i], lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


#   Gabor滤波过程
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


# Gabor特征提取
def get_gabor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filters = build_filters()
    res = []
    # matrix = []
    for i in range(len(filters)):
        gabor = process(img, filters[i])
        # matrix.append(np.array(gabor))
        gabor = gabor.flatten()
        # mean = np.mean(gabor)  # 均值
        # std = np.std(gabor)  # 标准差
        # var = np.var(gabor)  # 方差
        # skewness = np.mean((gabor - mean) ** 3)  # 偏度
        # kurtosis = np.mean((gabor - mean) ** 4) / pow(var, 2)  # 峰度
        D = pd.DataFrame(gabor)
        d = D.sum()  # 总和
        print(d)
        d = D.mean()  # 均值
        print(d)
        d = D.std()  # 标准差
        print(d)
        d = D.var()  # 方差
        print(d)
        d = D.corr(method='pearson')  # 相关系数
        print(d)
        d = D.corr(method='spearman')  # 相关系数
        print(d)
        d = D.corr(method='kendall')  # 相关系数
        print(d)
        d = D.cov()  # 协方差
        print(d)
        d = D.skew()  # 偏度
        print(d)
        d = D.kurt()  # 峰度
        print(d)
        d = D.describe()  # 属性统计
        print(d)
        return
        # res.append(mean)
        # res.append(std)
        # res.append(var)

    # print(res)
    # pl.figure(2)
    # for temp in range(len(matrix)):
    #     pl.subplot(4, 5, temp + 1)
    #     pl.imshow(matrix[temp], cmap='gray')
    # pl.show()

    return res


# 获取图像的纹理特征
def get_features(img):
    img = locate.replace_bg(img)
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    # gray = cv2.equalizeHist(gray)
    # 256阶化为64个等级
    # gray = (gray / 4).astype(np.int32)

    features = []

    # 计算 统计特性
    mean = np.mean(gray)  # 均值
    var = np.var(gray)  # 方差
    # std = np.std(gray)  # 标准差
    skewness = np.mean((gray - mean) ** 3)  # 偏度
    kurtosis = np.mean((gray - mean) ** 4) / pow(var, 2)  # 峰度
    # print(mean, std, skewness, kurtosis)
    features.append(mean)
    features.append(var)
    # features.append(std)
    features.append(skewness)
    features.append(kurtosis)

    # 提取 GLCM 特征（对比度、相异性、同质性、能量、相关性、角二阶矩）
    glcm = feature.greycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True,
                                normed=True)
    feature_types = {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}
    for feature_type in feature_types:
        fea = feature.greycoprops(glcm, feature_type)
        # features.append(np.mean(fea))
        features.append(fea[0][0])
        features.append(fea[0][1])
        features.append(fea[0][2])
        features.append(fea[0][3])

    # 提取 LBP 特征
    """
    'default'：原始的局部二值模式，它是灰度但不是旋转不变的。
    'ror'：扩展灰度和旋转不变的默认实现。
    'uniform'：改进的旋转不变性和均匀的模式以及角度空间的更精细的量化，灰度和旋转不变。
    'nri_uniform'：非旋转不变的均匀图案变体，它只是灰度不变的R199。
    'VAR'：局部对比度的旋转不变方差度量，图像纹理是旋转但不是灰度不变的。
    """
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    features.extend(hist)

    # 获取 Gabor 特征
    # filters = build_filters()
    # gabors = get_gabor(gray, filters)
    # features.extend(gabors)

    features = np.array(features)

    return features


if __name__ == '__main__':
    img = cv2.imread(r"F:\DataSet\testImages\bazidilaohu1#\sanjiao2-8.jpg")
    img = cv2.resize(img, (200, 200))
    # img = locate.replace_bg(img)
    # cv2.imshow("img", img)
    # res = get_features(img, "1")
    # print(res)
    get_gabor(img)
    # print(res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
