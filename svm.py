import os
import cv2
import numpy as np
import pandas as pd
from sklearn import svm
import joblib
import features
import time


# 获取训练数据和标签
def get_training_data_and_labels(path, label, name1, name2):
    for i in os.listdir(path):
        train_data = []
        train_labels = []
        img = cv2.imread(os.path.join(path, i))
        img = cv2.resize(img, (200, 200))
        feature = features.get_all_features(img)
        if feature is None:
            return
        train_data.append(feature)
        train_labels.append(label[0])
        data = pd.DataFrame(train_data)
        data.to_csv('D:\\DataSet\\svm_training_csv\\' + name1, mode='a', header=None, index=None)
        label = pd.DataFrame(train_labels)
        label.to_csv('D:\\DataSet\\svm_training_csv\\' + name2, mode='a', header=None, index=None)


# 模型训练
def svm_train(train_data, train_labels, test_data, test_labels):
    # 创建svm模型
    svm = cv2.ml.SVM_create()
    # 设置类型为SVM_C_SVC代表分类
    svm.setType(cv2.ml.SVM_C_SVC)
    # 设置核函数
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # 设置其它属性
    svm.setC(1)
    svm.setGamma(3)
    svm.setDegree(3)
    # 设置迭代终止条件
    svm.setTermCriteria((cv2.TermCriteria_MAX_ITER, 300, 1e-3))
    # 训练
    svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    svm.save('svm.xml')
    # 在测试数据上计算准确率
    # 进行模型准确率的测试 结果是一个元组 第一个值为数据1的结果
    test_pre = svm.predict(test_data)
    test_ret = test_pre[1]

    # 计算准确率
    test_ret = test_ret.reshape(-1, )
    test_labels = test_labels.reshape(-1, )
    test_sum = (test_ret == test_labels)
    acc = test_sum.mean()
    print(acc)


# 预测
def svm_predict(img):
    img_sw = img.copy()

    # 将数据类型由uint8转为float32
    img = img.astype(np.float32)
    # 图片转一维
    img = img.reshape(-1, )
    # 增加一个维度
    img = img.reshape(1, -1)
    # 图片数据归一化
    img = img / 255

    # 载入svm模型
    svm = cv2.ml.SVM_load('svm.xml')
    # 进行预测
    img_pre = svm.predict(img)
    print(img_pre[1])

    cv2.imshow('test', img_sw)
    cv2.waitKey(0)


# 计算特征
def make_features():
    file1 = r'D:\DataSet\svm_training_csv\train.csv'
    file2 = r'D:\DataSet\svm_training_csv\trainLabel.csv'
    file3 = r'D:\DataSet\svm_training_csv\test.csv'
    file4 = r'D:\DataSet\svm_training_csv\testLabel.csv'
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
    train_path = r"D:\DataSet\svm_training_imgs1"
    for i in os.listdir(train_path):
        filepath = os.path.join(train_path, i)
        get_training_data_and_labels(filepath, i, 'train.csv', 'trainLabel.csv')

    test_path = r"D:\DataSet\svm_test_imgs1"
    for j in os.listdir(test_path):
        filepath2 = os.path.join(test_path, j)
        get_training_data_and_labels(filepath2, j, 'test.csv', 'testLabel.csv')
    end = time.time()
    minute = (end - start) / 60
    print("计算特征耗时：", minute, "分钟")


# 加载特征
def load_features():
    train_data = np.array(pd.read_csv('D:\\DataSet\\svm_training_csv\\train.csv', header=None))
    train_label = np.array(pd.read_csv('D:\\DataSet\\svm_training_csv\\trainLabel.csv', header=None))
    test_data = np.array(pd.read_csv('D:\\DataSet\\svm_training_csv\\test.csv', header=None))
    test_label = np.array(pd.read_csv('D:\\DataSet\\svm_training_csv\\testLabel.csv', header=None))
    # train_data = np.array(train_data, dtype='float32')
    # test_data = np.array(test_data, dtype='float32')
    # print(test_data)
    # print(test_label)
    # svm_train(train_data, train_label, test_data, test_label)
    return train_data, train_label, test_data, test_label


# 训练
def train_model(train_data, train_label, test_data, test_label):
    print("开始训练")
    start = time.time()
    clf = svm.SVC(kernel='linear', C=1)
    # clf = svm.LinearSVC(max_iter=1000000000)
    # clf = svm.SVR(kernel='rbf', C=1.0, gamma='auto', degree=3)
    clf.fit(train_data, train_label.ravel())
    joblib.dump(clf, 'D:\\DataSet\\svm_training_csv\\svm2.joblib')
    end = time.time()
    minute = (end - start) / 60
    print("训练耗时：", minute, "分钟")
    print('训练集准确率：', clf.score(train_data, train_label.ravel()))
    print('测试集准确率：', clf.score(test_data, test_label.ravel()))


# 预测
def predict(path):
    img = cv2.imread(path)
    if img is None:
        return '未找到图片'
    img = cv2.resize(img, (200, 200))
    feature = features.get_all_features(img)
    clf = joblib.load("D:\\DataSet\\svm_training_csv\\svm.joblib")
    res = clf.predict([feature])
    return res


if __name__ == '__main__':
    # make_features()
    train_data, train_label, test_data, test_label = load_features()
    train_model(train_data, train_label, test_data, test_label)
    # path = r'D:\DataSet\svm_training_imgs\4\1 (57).jpg'
    # print(predict(path))
