import cv2
import numpy as np
import pylab as pl
import locate
import skimage.feature as feature
from skimage.feature import local_binary_pattern


# 构建gabor滤波器
def build_filters():
    filters = []
    # ksize = [7, 9, 11, 13, 15, 17]  # gabor尺度，6个
    ksize = [3, 5, 7, 9, 11]  # gabor尺度，5个
    lamda = np.pi / 2.0  # 波长
    theta = [0, np.pi / 2, np.pi / 4, np.pi - np.pi / 4]  # gabor方向，0°，45°，90°，135°，共四个
    for i in range(len(theta)):
        for K in range(len(ksize)):
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta[i], lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


# Gabor滤波过程
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


# Gabor特征提取
def get_gabor(img):
    # img = locate.replace_bg(img)
    img = locate.get_pest_img(img)
    if img is None:
        return None
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filters = build_filters()
    res = []
    # matrix = []
    for i in range(len(filters)):
        gabor = process(img, filters[i])
        # matrix.append(np.array(gabor))
        gabor = gabor.flatten()

        # sums = np.sum(gabor)  # 总和
        mean = np.mean(gabor)  # 均值
        std = np.std(gabor)  # 标准差
        var = np.var(gabor)  # 方差
        # corr = np.corrcoef(gabor)  # 相关系数
        cov = np.cov(gabor)  # 协方差
        skewness = np.mean((gabor - mean) ** 3)  # 偏度
        # kurtosis = np.mean((gabor - mean) ** 4) / pow(var, 2)  # 峰度

        # res.append(sums)
        # res.append(mean)
        res.append(std)
        res.append(var)
        # res.append(corr)
        res.append(cov)
        res.append(skewness)
        # res.append(kurtosis)

    # pl.figure(2)
    # for temp in range(len(matrix)):
    #     pl.subplot(4, 5, temp + 1)
    #     pl.imshow(matrix[temp], cmap='gray')
    # pl.show()

    return res


# 获取原图像的统计特征
def get_statistical_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    array = gray.flatten()
    # sums = np.sum(array)  # 总和
    mean = np.mean(array)  # 均值
    # var = np.var(array)  # 方差
    std = np.std(array)  # 标准差
    cov = np.cov(array)  # 协方差
    skewness = np.mean((array - mean) ** 3)  # 偏度
    # kurtosis = np.mean((array - mean) ** 4) / pow(var, 2)  # 峰度
    # corr = np.corrcoef(array)  # 相关系数
    res = []
    # res.append(sums)
    res.append(mean)
    # res.append(var)
    res.append(std)
    res.append(cov)
    res.append(skewness)
    # res.append(kurtosis)
    # res.append(corr)
    return res


# 获取 GLCM 特征
def get_glcm(img):
    # img = locate.get_pest_img(img)
    img = locate.replace_bg(img)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = []
    # 提取 GLCM 特征（对比度、相异性、同质性、能量、相关性、角二阶矩）
    glcm = feature.greycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True,
                                normed=True)
    feature_types = {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}
    for feature_type in feature_types:
        fea = feature.greycoprops(glcm, feature_type)
        # features.append(np.mean(fea))
        res.append(fea[0][0])
        res.append(fea[0][1])
        res.append(fea[0][2])
        res.append(fea[0][3])
    # print(len(res))
    return res


# 获取 LBP 特征
def get_lbp(img):
    img = locate.get_pest_img(img)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """
    'default'：原始的局部二值模式，它是灰度但不是旋转不变的。
    'ror'：扩展灰度和旋转不变的默认实现。
    'uniform'：改进的旋转不变性和均匀的模式以及角度空间的更精细的量化，灰度和旋转不变。
    'nri_uniform'：非旋转不变的均匀图案变体，它只是灰度不变的R199。
    'VAR'：局部对比度的旋转不变方差度量，图像纹理是旋转但不是灰度不变的。
    """
    radius = 1
    n_points = 8 * radius
    # lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    # (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # hist = hist.astype("float")
    # hist /= (hist.sum() + 1e-7)

    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    max_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    return hist


# 获取图像的纹理特征
def get_all_features(img):
    features = []
    statistic = get_statistical_features(img)
    # lbp = get_lbp(img)
    glcm = get_glcm(img)
    gabor = get_gabor(img)
    if gabor is None or glcm is None:
        return None

    features.extend(statistic)
    # features.extend(lbp)
    features.extend(glcm)
    features.extend(gabor)

    # features = np.array(features)

    return features


if __name__ == '__main__':
    img = cv2.imread(r"D:\DataSet\svm_training_imgs\1\1 (2).jpg")
    img = cv2.resize(img, (200, 200))
    # img = locate.replace_bg(img)
    # cv2.imshow("img", img)
    # res = get_features(img, "1")
    # print(res)
    get_glcm(img)
    # print(res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
