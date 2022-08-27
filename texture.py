import sys
import svm


def texture_classify(path):
    res = svm.predict(path, "texture")
    res = str(res)
    res = res[2:-2]
    if res == 'under':
        return '仰姿'
    elif res == '0':
        return '八字地老虎'
    elif res == '1':
        return '扁刺蛾'
    elif res == '2':
        return '甘蓝夜蛾'
    elif res == '3':
        return '甘暮天蛾'
    elif res == '4':
        return '黄地老虎'
    elif res == '5':
        return '金龟子'
    elif res == '6':
        return '愧尺蠖'
    elif res == '7':
        return '六点天蛾'
    elif res == '8':
        return '棉铃虫'
    elif res == '9':
        return '粘虫'
    elif res == '10':
        return '深色白眉天蛾'
    elif res == '11':
        return '小地老虎'
    elif res == '12':
        return '银锭夜蛾'
    elif res == '13':
        return '榆绿天蛾'
    elif res == '14':
        return '未定种'
    elif res == '15':
        return '美国白蛾'
    elif res == '16':
        return '旋幽夜蛾1'
    elif res == '17':
        return '旋幽夜蛾2'


if __name__ == '__main__':
    for i in range(1, len(sys.argv)):
        print(texture_classify(sys.argv[i]))
    # path = r'F:\DataSet\svm_train_imgs\1\1 (4).jpg'
    # print(texture_classify(path))
