import sys
import svm


def texture_classify(path):
    return svm.predict(path, "texture")


if __name__ == '__main__':
    # for i in range(1, len(sys.argv)):
    #     print(texture_classify(sys.argv[i]))
    path = r'F:\DataSet\contour_train_imgs\changtiao\ct (26).jpg'
    print(texture_classify(path))
