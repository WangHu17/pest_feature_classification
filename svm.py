import os

import cv2
import glcm
import numpy as np


#获取训练数据和标签
def get_training_data_and_labels(trainData, trainLabels, path, label):
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i))
        res = glcm.get_glcm(img)
        
