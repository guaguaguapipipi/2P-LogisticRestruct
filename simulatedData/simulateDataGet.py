# 模拟数据生成文件
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.optimize import curve_fit


#%%
#P0 r K
def funclog(X, c0, r, Cmax):
    return Cmax * c0 * np.exp(r * X) / (Cmax + c0 * (np.exp(r * X) - 1))

def fit_L2func(t, c1, c2):
    return 1 / (1 + (c1 * np.exp((-c2) * t)))



def getL2Original(dateArr, vOriArr):
    vOriArr = [float(i) for i in vOriArr]
    vOriArr = np.array(vOriArr)
    RLCArr = 1-(np.log(vOriArr[1:])-np.log(vOriArr[:-1]))
    minR = np.min(RLCArr)
    NRLCQ = (RLCArr - minR) / (1 - minR)
    popt, pcov = curve_fit(fit_L2func, dateArr[1:], NRLCQ,
                           p0=np.array([1000, 0.01]), maxfev=20000000)
    NRLCQ_fit = fit_L2func(dateArr[1:], popt[0], popt[1])
    fitclist = np.exp(1 - (NRLCQ_fit * (1 - minR) + minR) + np.log(vOriArr[:-1]))
    fitclist = np.insert(fitclist, 0, vOriArr[0])
    return fitclist, popt


def getVirtualData(datelen, randRate):
    c1 = 100 #1000
    c3 = 100000 #10000
    c2 = 0.1  # 0.1,这个就是r值
    dateArr = np.array([i for i in range(datelen)])  # 这个是生成的日期
    vOriArr = funclog(dateArr, c1, c2, c3)  # 这个是生成的原始数据
    L2vOriArr, popt = getL2Original(dateArr, vOriArr)  # 这个是得到L2的模拟参数
    incVirArr = vOriArr.copy()
    incVirArr[0] = incVirArr[0] - c1
    incVirArr[1:] = incVirArr[1:] - incVirArr[:-1]  # 这个是没加噪音前的增长值数据
    for i in range(len(incVirArr)):
        incVirArr[i] = incVirArr[i] * (1 + random.uniform(-randRate, randRate))
    finVirArr = incVirArr.copy()  # 这个是还原之后的值
    finVirArr[0] += c1
    for i in range(1, len(finVirArr)):
        finVirArr[i] += finVirArr[i-1]
    plt.plot(dateArr, finVirArr, 'r')
    # L2vOriArr, popt = getL2Original(dateArr, finVirArr)  # 这个是得到L2的模拟参数
    plt.plot(dateArr, vOriArr, 'r', color='blue')
    plt.plot(dateArr, L2vOriArr, 'r', color='green')
    plt.xlabel('time')
    plt.ylabel('CCC')
    plt.show()
    filename = "virtualData.csv"
    with open(filename, 'a+') as f:
        csv_write = csv.writer(f)
        infoStr = "popt={}^^^{}&c1={},c2={},c3={},randRate={}".format(popt[0], popt[1], c1, c2, c3, randRate)
        data_row = [infoStr] + list(finVirArr)
        csv_write.writerow(data_row)
        f.close
    return 1

#%%

datelen = 150
filename = "virtualData.csv"
dateList = [i for i in range(datelen)]
datestr = "Country/Region"
with open(filename, 'a+') as f:
    csv_write = csv.writer(f)
    data_row = [datestr] + dateList
    csv_write.writerow(data_row)
    f.close
for i in [0.25, 0.5, 0.75, 0.9, 1]:
    getVirtualData(datelen, i)


