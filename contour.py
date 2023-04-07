import sys
import os
import svm
import time


def contour_classify(path):
    res = svm.predict(path, "contour")
    res = str(res)
    res = res[2:-2]
    if res == 'sanjiao':
        return '三角形'
    elif res == 'changtiao':
        return '长条形'
    elif res == 'xiaodian':
        return '小点形'
    elif res == 'diechi':
        return '蝶翅形'
    elif res == 'dachi':
        return '大翅形'
    elif res == 'zhanchi':
        return '展翅形'
    elif res == 'yi':
        return '异形'


if __name__ == '__main__':
    # for i in range(1, len(sys.argv)):
    #     print(contour_classify(sys.argv[i]))
    # path = r'F:\DataSet\contour_train_imgs\changtiao\ct (2).jpg'
    # print(contour_classify(path))
    print("程序正在运行...")
    T1 = time.time()
    root = "F:\\DataSet\\testImages1"
    dirs = os.listdir(root)
    for dir in dirs:
        path = root + "\\" + dir
        files = os.listdir(path)
        index = 0
        for file in files:
            index = index + 1
            filePath = root + "\\" + dir + "\\" + file
            resColor = contour_classify(filePath)
            newName = root + "\\" + dir + "\\" + resColor + str(index) + ".jpg"
            os.rename(filePath, newName)
    T2 = time.time()
    print('程序运行时间为: {:.2f}秒'.format(T2 - T1))
