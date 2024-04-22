import random

from resultFile import rightXicData, errorXICData
import PRight20RankData0 as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def checkData():
    for i in rightXicData.xicList:
        print(len(i))
        for j in i:
            if(len(j)!=16):
                print(len(j))
    print("---------------")
    for i in errorXICData.xicList:
        print(len(i))
        for j in i:
            if(len(j)!=16):
                print(len(j))

def getOneXicMap(country, datefirst, datelast):
    country -= 1
    plt.figure()
    datalen = len(errorXICData.xicList)
    x = [i for i in range(5, 21)]

    plt.subplot(2,1,1)
    plt.title("RealPredict",loc="right")
    plt.xlabel('windowSize')
    plt.ylabel('XIC')
    count = 1
    for j in errorXICData.xicList[country][datefirst:datelast]:
        cl = ''.join(random.choice("1234567890ABCDEF") for _ in range(6))
        plt.plot(x, j, 'r', color="#"+cl, label=str(count))
        plt.plot(x[j.index(min(j))], min(j), 's', color="#" + cl)
        count += 1
    plt.legend(loc=0)  # 指定legend的位置右下角

    plt.subplot(2, 1, 2)
    plt.title("RealSample",loc="right")
    plt.xlabel("windowSize")
    plt.ylabel("XIC")
    count = 1
    for j in rightXicData.xicList[country][datefirst:datelast]:
        cl = ''.join(random.choice("1234567890ABCDEF") for _ in range(6))
        plt.plot(x, j, 'r', color="#" + cl, label=str(count))
        plt.plot(x[j.index(min(j))], min(j), 's', color="#" + cl)
        count += 1
    plt.legend(loc=0)

    plt.show()

def getXicDistribution():
    allCountMinXic = []
    for i in dt.xicList:
        countMinXIC = []
        for j in i:
            countMinXIC.append(j.index(min(j))+5)
        allCountMinXic.append(countMinXIC)
    allNumList = []
    for i in allCountMinXic:
        numList = [0 for i in range(21)]
        for j in i:
            numList[j] += 1
        allNumList.append(numList)
    allNumArray = np.array(allNumList).T[5:]
    pd_data = pd.DataFrame(allNumArray,columns=[155, 21, 71, 98, 153, 77, 56, 73, 138, 124])
    print(pd_data)
    pd_data.to_csv("WightXicDistribution.csv")



getXicDistribution()


# for i in range(0, 50):
#     getOneXicMap(1, 40+i*5, 45+i*5)
