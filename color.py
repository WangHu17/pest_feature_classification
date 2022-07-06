import os
import sys
import cv2
import kmeans


def color_classify(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (600, 600))
    cv2.imshow("color", img)
    hsv_colors, clusters = kmeans.get_main_colors(img)
    main_colors = []
    main_index = []
    min = 1
    minindex = -1
    # 获取两个主色
    for i in range(0, len(clusters)):
        if clusters[i] < min:
            min = clusters[i]
            minindex = i
    for i in range(0, len(hsv_colors)):
        if i != minindex and (hsv_colors[i] != [60, 255, 255]).any() and (hsv_colors[i] != [60, 255, 254]).any() and (hsv_colors[i] != [60, 254, 255]).any():
            main_colors.append(hsv_colors[i])
            main_index.append(i)
    # 主色排序
    if clusters[main_index[0]] < clusters[main_index[1]]:
        temp = main_colors[0]
        main_colors[0] = main_colors[1]
        main_colors[1] = temp
    # print(main_colors[0], main_colors[1])
    # 判断颜色
    resColor = ""
    for i in range(0, len(main_colors)):
        h = main_colors[i][0]
        s = main_colors[i][1]
        v = main_colors[i][2]
        if s <= 10:
            if resColor.find("白") == -1:
                resColor += "白"
        elif 20 <= h <= 22 and 28 <= s <= 85 and v >= 128:
            if resColor.find("黄") == -1:
                resColor += "淡黄"
        # elif 17 <= h <= 19 and 108 <= s <= 150 and 81 <= v <= 91:
        #     if resColor.find("棕") == -1:
        #         resColor += "棕"
        elif 17 <= h <= 23 and 76 <= s <= 147 and 100 <= v <= 146:
            if resColor.find("黄") == -1:
                resColor += "黄"
        elif 18 <= h <= 21 and 63 <= s <= 85 and 80 <= v <= 100:
            if resColor.find("灰") == -1:
                resColor += "灰"
        elif 11 <= h <= 16 and 145 <= s <= 188 and 34 <= v <= 68:
            if resColor.find("棕") == -1:
                resColor += "棕"
        elif 15 <= h <= 19 and 115 <= s <= 150 and 34 <= v <= 50:
            if resColor.find("棕") == -1:
                resColor += "棕"
        elif v <= 46:
            if resColor.find("黑") == -1:
                resColor += "黑"
        elif 9 <= h <= 15 and 66 <= s <= 91 and 66 <= v <= 179:
            if resColor.find("土褐") == -1:
                resColor += "土褐"
        elif 17 <= h <= 20 and 51 <= s <= 73 and 90 <= v <= 130:
            if resColor.find("绿") == -1:
                resColor += "绿"
        elif 14 <= h <= 23 and 68 <= s <= 168 and 47 <= v <= 102:
            if resColor.find("褐") == -1:
                resColor += "褐"
    return resColor


if __name__ == '__main__':
    for i in range(1, len(sys.argv)):
        print(color_classify(sys.argv[i]))
    # 识别个体
    # path = "F:\\DataSet\\baseImgs\\die2.jpg"
    # print(color_classify(path))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 识别一个种类
    # dirs = ["huangdilaohu1", "huangdilaohu2", "huangdilaohu3", "xuanyouyee5", "xuanyouyee6"]
    # for i in range(4):
    # dir = "F:\\DataSet\\testImages\\yulvtiane1#"
    # files = os.listdir(dir)
    # index = 0
    # for file in files:
    #     index = index + 1
    #     filePath = dir + "\\" + file
    #     resColor = color_classify(filePath)
    #     newName = dir + "\\" + resColor + str(index) + ".jpg"
    #     os.rename(filePath, newName)

    # 识别指定图像
    # dir = "F:\\DataSet\\testImages\\huangdilaohu1"
    # for i in range(124):
    #     filePath = dir + "\\1 (" + str(i+1) + ").jpg"
    #     resColor = color_classify(filePath)
    #     newName = dir + "\\1" + resColor + str(i+1) + ".jpg"
    #     os.rename(filePath, newName)

    # 识别所有种类
    # root = "F:\\DataSet\\baseImages"
    # dirs = os.listdir(root)
    # for dir in dirs:
    #     path = root + "\\" + dir
    #     files = os.listdir(path)
    #     index = 0
    #     for file in files:
    #         index = index + 1
    #         filePath = root + "\\" + dir + "\\" + file
    #         newName = root + "\\" + dir + "\\" + str(index) + ".jpg"
    #         os.rename(filePath, newName)
    # for dir in dirs:
    #     path = root + "\\" + dir
    #     files = os.listdir(path)
    #     index = 0
    #     for file in files:
    #         index = index + 1
    #         filePath = root + "\\" + dir + "\\" + file
    #         resColor = color_classify(filePath)
    #         newName = root + "\\" + dir + "\\" + resColor + str(index) + ".jpg"
    #         os.rename(filePath, newName)
