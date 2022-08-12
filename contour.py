import sys
import svm


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
    for i in range(1, len(sys.argv)):
        print(contour_classify(sys.argv[i]))
    # path = r'F:\DataSet\contour_train_imgs\changtiao\ct (2).jpg'
    # print(contour_classify(path))
