import os
import cv2
import numpy as np
from skimage.feature import greycomatrix as gm
from skimage.feature import greycoprops as gp
import random as rd
import pandas as pd
from sklearn import svm


def get_feature(file):
    feature = []
    im = cv2.imread(file, 0)
    m = gm(im, [10], [0, np.pi / 2], levels=256, symmetric=True)
    con = gp(m, 'contrast')
    feature.append(con[0, 0])
    feature.append(con[0, 1])

    dis = gp(m, 'dissimilarity')
    feature.append(dis[0, 0])
    # feat
    feature.append(dis[0, 1])

    hom = gp(m, 'homogeneity')
    feature.append(hom[0, 0])
    feature.append(hom[0, 1])

    ene = gp(m, 'energy')
    feature.append(ene[0, 0])
    feature.append(ene[0, 1])

    cor = gp(m, 'correlation')
    feature.append(cor[0, 0])
    feature.append(cor[0, 1])

    asm = gp(m, 'ASM')
    feature.append(asm[0, 0])
    feature.append(asm[0, 1])

    df = np.array(feature)

    return df


def get_xy(file, savex, savey):
    pic_l = os.listdir(file)
    rd.shuffle(pic_l)
    for pic in pic_l:
        # print('getting')
        # print(pic)
        x = []
        y = []
        if pic[-3:] == 'jpg':
            x_raw = get_feature(file + '\\' + pic)
            x.append(x_raw)
            y.append(pic[-5])
            x = np.array(x)
            y = np.array(y)
            xtr_df = pd.DataFrame(x)
            xtr_df.to_csv('F:\\DataSet\\svm_training_csv\\' + savex, mode='a', header=None, index=None)
            ytr_df = pd.DataFrame(y)
            ytr_df.to_csv('F:\\DataSet\\svm_training_csv\\' + savey, mode='a', header=None, index=None)

    return pic_l


if __name__ == '__main__':
    file1 = r'F:\DataSet\svm_training_csv\x_train.csv'
    file2 = r'F:\DataSet\svm_training_csv\y_train.csv'
    file3 = r'F:\DataSet\svm_training_csv\x_test.csv'
    file4 = r'F:\DataSet\svm_training_csv\y_test.csv'
    os.remove(file1)
    os.remove(file2)
    os.remove(file3)
    os.remove(file4)
    get_xy(r'F:\DataSet\svm_training_csv\train', 'x_train.csv', 'y_train.csv')
    get_xy(r'F:\DataSet\svm_training_csv\test', 'x_test.csv', 'y_test.csv')
    x_train = np.array(pd.read_csv(r'F:\DataSet\svm_training_csv\x_train.csv', header=None))
    y_train = np.array(pd.read_csv(r'F:\DataSet\svm_training_csv\y_train.csv', header=None))
    x_test = np.array(pd.read_csv(r'F:\DataSet\svm_training_csv\x_test.csv', header=None))
    y_test = np.array(pd.read_csv(r'F:\DataSet\svm_training_csv\y_test.csv', header=None))
    # print(x_test)
    # print(y_test)
    clf = svm.SVC(kernel='linear', C=0.7, degree=4)
    clf.fit(x_train, y_train.ravel())
    print(clf.score(x_test, y_test.ravel()))