import os
import cv2
import numpy as np
import pandas as pd
from sklearn import svm
import joblib

import contour_features
import texture_features
import time


# 获取训练数据和标签
def get_training_data_and_labels(type, path, label, name1, name2):
    label1 = label
    for i in os.listdir(path):
        train_data = []
        train_labels = []
        img = cv2.imread(os.path.join(path, i))
        img = cv2.resize(img, (200, 200))
        feature = None
        if type == 'texture':
            feature = texture_features.get_all_features(img)
        elif type == 'contour':
            feature = contour_features.get_contour_features(img)
        if feature is None:
            print(label1, i)
            continue
        train_data.append(feature)
        train_labels.append(label1)
        data = pd.DataFrame(train_data)
        data.to_csv('F:\\DataSet\\svm_training_csv\\' + type + '\\' + name1, mode='a', header=None, index=None)
        label = pd.DataFrame(train_labels)
        label.to_csv('F:\\DataSet\\svm_training_csv\\' + type + '\\' + name2, mode='a', header=None, index=None)


# 计算特征
def make_features(type, train_path, test_path):
    root = 'F:\\DataSet\\svm_training_csv\\'
    file1 = root + type + '\\train.csv'
    file2 = root + type + '\\trainLabel.csv'
    file3 = root + type + '\\test.csv'
    file4 = root + type + '\\testLabel.csv'
    if os.path.exists(file1):
        os.remove(file1)
    if os.path.exists(file2):
        os.remove(file2)
    if os.path.exists(file3):
        os.remove(file3)
    if os.path.exists(file4):
        os.remove(file4)

    print("开始计算特征")
    start = time.time()
    for i in os.listdir(train_path):
        filepath = os.path.join(train_path, i)
        get_training_data_and_labels(type, filepath, i, 'train.csv', 'trainLabel.csv')

    for j in os.listdir(test_path):
        filepath2 = os.path.join(test_path, j)
        get_training_data_and_labels(type, filepath2, j, 'test.csv', 'testLabel.csv')
    end = time.time()
    minute = (end - start) / 60
    print("计算特征耗时：", minute, "分钟")


# 加载特征
def load_features(type):
    train_data = np.array(pd.read_csv('F:\\DataSet\\svm_training_csv\\' + type + '\\train.csv', header=None))
    train_label = np.array(pd.read_csv('F:\\DataSet\\svm_training_csv\\' + type + '\\trainLabel.csv', header=None))
    test_data = np.array(pd.read_csv('F:\\DataSet\\svm_training_csv\\' + type + '\\test.csv', header=None))
    test_label = np.array(pd.read_csv('F:\\DataSet\\svm_training_csv\\' + type + '\\testLabel.csv', header=None))
    return train_data, train_label, test_data, test_label


# 训练模型
def train_model(train_data, train_label, test_data, test_label, type):
    print("开始训练")
    start = time.time()
    clf = svm.SVC(kernel='linear', C=1)
    # clf = svm.LinearSVC(max_iter=1000000000)
    # clf = svm.SVR(kernel='rbf', C=1.0, gamma='auto', degree=3)
    clf.fit(train_data, train_label.ravel())
    joblib.dump(clf, 'F:\\DataSet\\svm_training_csv\\' + type + '.joblib')
    end = time.time()
    minute = (end - start) / 60
    print("训练耗时：", minute, "分钟")
    print('训练集准确率：', clf.score(train_data, train_label.ravel()))
    print('测试集准确率：', clf.score(test_data, test_label.ravel()))


# 预测
def predict(path, type):
    img = cv2.imread(path)
    if img is None:
        return '未找到图片'
    img = cv2.resize(img, (200, 200))
    feature = None
    if type == 'texture':
        feature = texture_features.get_all_features(img)
    elif type == 'contour':
        feature = contour_features.get_contour_features(img)
    if feature is None:
        return '识别失败'
    clf = joblib.load('F:\\DataSet\\svm_training_csv\\' + type + '.joblib')
    res = clf.predict([feature])
    return res


if __name__ == '__main__':
    train_path = r'F:\DataSet\svm_train_imgs'
    test_path = r'F:\DataSet\svm_test_imgs'
    type = 'texture'
    make_features(type, train_path, test_path)
    train_data, train_label, test_data, test_label = load_features(type)
    train_model(train_data, train_label, test_data, test_label, type)
    '''
    开始计算特征
    计算特征耗时： 3.3965014696121214 分钟
    开始训练
    训练耗时： 0.7903028845787048 分钟
    训练集准确率： 0.9332579185520362
    测试集准确率： 0.869281045751634
    '''
