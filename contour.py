import sys
import svm


def contour_classify(path):
    return svm.predict(path, "contour")


if __name__ == '__main__':
    for i in range(1, len(sys.argv)):
        print(contour_classify(sys.argv[i]))
    # path = r'F:\DataSet\contour_train_imgs\changtiao\ct (26).jpg'
    # print(contour_classify(path))
