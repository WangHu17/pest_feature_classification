import os
import cv2
import numpy as np
import glcm


# 获取训练数据和标签
def get_training_data_and_labels(train_data, train_labels, path, label):
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i))
        feature = glcm.get_glcm(img)
        train_data.append(feature)
        train_labels.append(label)
    return train_data, train_labels


# 模型训练
def svm_train(train_data, train_labels, test_data, test_labels):
    # 创建svm模型
    svm = cv2.ml.SVM_create()
    # 设置类型为SVM_C_SVC代表分类
    svm.setType(cv2.ml.SVM_C_SVC)
    # 设置核函数
    svm.setKernel(cv2.ml.SVM_POLY)
    # 设置其它属性
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


if __name__ == '__main__':
    train_path = r""