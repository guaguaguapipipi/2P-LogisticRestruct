#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
from math import sqrt, pow
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import scipy.stats as stats


# In[2]:


def getR2(real_pred_ccc, fit_pred_ccc):
    real_pred_ccc = list(real_pred_ccc)
    fit_pred_ccc = list(fit_pred_ccc)
    print("-----------------------------")
    print(real_pred_ccc)
    print(fit_pred_ccc)
    print("-----------------------------")
    ssr = 0
    sst = 0
    tot = 0
    for i in real_pred_ccc:
        tot += i
    avg = tot/len(real_pred_ccc)
    for i in fit_pred_ccc:
        ssr += pow((i-avg),2)
    for i in real_pred_ccc:
        sst += pow((i-avg),2)
    return ssr/sst


# In[3]:


def fit_function(t, c1, c2):
    return 1/(1+(c1*np.exp((-c2)*t)))


# In[4]:


def BIC2(Y, pred_Y, k):
    # pred_Y是预测值，Y是实际值，k是参数的数量
    Q = np.sum((np.array(Y)-np.array(pred_Y))**2)
    n = len(Y)
    lnL = -n/2*(np.log(2*np.pi*Q/n)+1)
    bic = -2*lnL+np.log(n)*k
    return bic


def HQIC(Y, pred_Y, k):
    # pred_Y是预测值，Y是实际值，k是参数的数量
    Q = np.sum((np.array(Y)-np.array(pred_Y))**2)
    n = len(Y)
    lnL = -n/2*(np.log(2*np.pi*Q/n)+1)
    bic = -2*lnL+np.log(np.log(n))*k
    return bic


# In[5]:


def AIC(Y, pred_Y, k):
    # pred_Y是预测值，Y是实际值，k是参数的数量
    Q = np.sum((np.array(Y)-np.array(pred_Y))**2)
    n = len(Y)
    lnL = -n/2*(np.log(2*np.pi*Q/n)+1)
    bic = 2*k-2*lnL
    return bic


def QAIC(Y, pred_Y, k):
    # pred_Y是预测值，Y是实际值，k是参数的数量
    Q = np.sum((np.array(Y)-np.array(pred_Y))**2)
    n = len(Y)
    lnL = -n/2*(np.log(2*np.pi*Q/n)+1)
    bic = 2*k-2*lnL*(1-r2_score(Y, pred_Y))
    return bic

# In[6]:


def getAllDataFromCsvFile():
    while 1:
        try:
            filename = '51419-global_data.csv'
            datadict = {}
            with open(filename, encoding='utf-8-sig') as csvfile:
                csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                # header = next(csv_reader)        # 读取第一行每一列的标题
                for row in csv_reader:            # 将csv 文件中的数据保存到data中
                    datadict[row[0]] = row[1:]
                csvfile.close()
            break
        except:
            continue
    return datadict


# In[7]:


def selectOneCountry(selectid):
    datadict = getAllDataFromCsvFile()
    countlist = []
    for key in datadict.keys():
        if key == 'Country':
            continue
        countlist.append(key)
    datelist = datadict['Country']
    ccclist = datadict[countlist[selectid]]
    ccclist = [float(x) for x in ccclist]
    return datelist, ccclist


# In[8]:


def clearOData(datelist, ccclist):
    for i in range(len(ccclist)):
        if ccclist[i] != 0:
            datelist = datelist[i:]
            ccclist = ccclist[i:]
            break
    return datelist, ccclist


# In[9]:


def analysisNRCLQ(RLCQarray, MIN):
    """
    如果是预测的时候调用,那MIN要设定为None,如果是获取实际值的时候就要在MIN中放入前段参考值中的最小值
    """
    if MIN is None:
        MIN = min(RLCQarray)
    NRLCQarray = (RLCQarray-MIN)/(1-MIN)
    return NRLCQarray


# In[10]:


def getDateAndRLCQ(selectcountid):
    # ccc是指那个总的ccc列表，cccT是指t天的ccc，cccT1是指t-1天的ccc，两个数组是错开的，一个去掉了第一天的，一个去掉了最后一天的
    datelist, ccclist = selectOneCountry(selectcountid)
    datelist, ccclist = clearOData(datelist, ccclist)
    cccTlist = [i for i in ccclist]
    cccTlist.pop(0)
    datelist.pop(0)
    cccTarray = np.array(cccTlist)
    cccT1list = [i for i in ccclist]
    cccT1list.pop(len(cccT1list)-1)
    cccT1array = np.array(cccT1list)
    RLCQarray = 1-(np.log(cccTarray) - np.log(cccT1array))
    ccclist.pop(0)
    return datelist, RLCQarray, ccclist


# In[11]:


def getRLCQmap():
    datelist, RLCQarray, ccclist = getDateAndRLCQ()
    print(datelist)
    print(RLCQarray)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xdatelist = [i for i in range(0, len(RLCQarray))]
    ax.set(xlim=[0, max(xdatelist)],
           ylim=[min(RLCQarray), max(RLCQarray)],
           ylabel='RLCCCQ',
           xlabel='date')
    plt.plot(xdatelist, RLCQarray, color='lightblue', linewidth=3)
    # plt.scatter(xdatelist, LCQarray, color='lightblue', linewidth=3)
    return 1
# getRLCQmap()


# In[12]:


def fitNRLCQ(RLCQarray, selectIDX, i, pred_len):
    RLCQkeyarray = RLCQarray[(selectIDX-i+1):selectIDX+1]  # 参考RLCQ值的array
    NRLCQarray = analysisNRCLQ(RLCQkeyarray, None)  # 参考NRLCQ值的array
    datelist = np.array([j for j in range(len(NRLCQarray))])  # 参考date值的list
    popt, pcov = curve_fit(fit_function, datelist, NRLCQarray, maxfev=2000000)
    fit_NRLCQ = fit_function(np.array(datelist), popt[0], popt[1])
    pred_datearray = np.array([j for j in range(len(NRLCQarray),
                                                len(NRLCQarray)+pred_len)])
    pred_NRLCQ = fit_function(pred_datearray, popt[0], popt[1])
    allRLCQarray = RLCQarray[(selectIDX-i+1):selectIDX+pred_len+1]
    allNRLCQarray = analysisNRCLQ(allRLCQarray, min(RLCQkeyarray))
    # xic = HQIC(NRLCQarray, fit_NRLCQ, 2)
    # xic = BIC2(NRLCQarray, fit_NRLCQ, 2)
    xic = AIC(NRLCQarray, fit_NRLCQ, 2)
    # xic = AIC(allNRLCQarray[(len(allNRLCQarray)-len(pred_NRLCQ)):], pred_NRLCQ, 2)
    # xic = abs(aic) + abs(bic)
    # if xic < 0:
    #     xic = 20000
    return xic, NRLCQarray, fit_NRLCQ, pred_NRLCQ, allNRLCQarray, datelist, pred_datearray


# In[13]:


def mapPredictNRL(NRLCQarray, fit_NRLCQ, pred_NRLCQ, allNRLCQarray, datelist, pred_datearray):
    plot1 = plt.plot(datelist, NRLCQarray, 's', label="NRLCQSample")
    plot2 = plt.plot(datelist, fit_NRLCQ, 'r', label='predict fit NRLCQ')
    plot3 = plt.plot(pred_datearray, pred_NRLCQ, 'r', label='predict future NRLCQ')
    plot4 = plt.plot(pred_datearray, allNRLCQarray[len(datelist):], 's', label="real futrure NRLCQ")
    plt.xlabel('time')
    plt.ylabel('NRLCCCQ')
    plt.legend(loc=0)  # 指定legend的位置右下角
    return 1


# In[14]:


def getPredictCCC(RLCQarray, NRLCQarray, pred_NRLCQ, selectIDX, i, pred_len, allCCCtArray):
    MIN = min(RLCQarray[(selectIDX-i+1):selectIDX+1])
    ENarray = np.exp(1-(pred_NRLCQ*(1-MIN))-MIN)
    CCCt1 = allCCCtArray[i-1]
    predictCCClist = []
    for i in range(len(ENarray)):
        CCCt1 = ENarray[i]*CCCt1
        predictCCClist.append(CCCt1)
    return predictCCClist


# In[15]:


def predictNRLCQ(pred_len):
    datelist, RLCQarray, ccclist = getDateAndRLCQ()
    for i in range(len(datelist)):
        print(str(i) + ' : ' + datelist[i], end=', ')
    selectIDX = int(input("Select an index of date"))
    xicmin = 20000
    for i in range(3, 200):
        xic, NRLCQarray, fit_NRLCQ, pred_NRLCQ, allNRLCQarray, datelist, pred_datearray = fitNRLCQ(RLCQarray, selectIDX, i, pred_len)
        print(str(i) + " : " + str(xic))
        if xic > xicmin:
            i -= 1
            break
        else:
            xicmin = xic
    xic, NRLCQarray, fit_NRLCQ, pred_NRLCQ, allNRLCQarray, datelist, pred_datearray = fitNRLCQ(RLCQarray, selectIDX, i, pred_len)
    print(str(i))
    print(xic)
    print(NRLCQarray)
    print(fit_NRLCQ)
    print(pred_NRLCQ)
    print(allNRLCQarray)
    print(allNRLCQarray[len(fit_NRLCQ):])
    print(pred_NRLCQ)
    mapPredictNRL(NRLCQarray, fit_NRLCQ, pred_NRLCQ, allNRLCQarray, datelist, pred_datearray)
    return 1


# In[16]:


def mapPredictCCC(allCCCtArray, predictCCClist):
    datelist = [i for i in range(len(allCCCtArray))]
    # print(datelist)
    # print(datelist[(len(allCCCtArray)-len(predictCCClist)):])
    print(allCCCtArray)
    print(predictCCClist)
    plot1 = plt.plot(datelist, allCCCtArray, 's', label="realCCC")
    plot2 = plt.plot(datelist[(len(allCCCtArray)-len(predictCCClist)):], predictCCClist, 's', label='predictCCC')
    plt.xlabel('time')
    plt.ylabel('CCC')
    plt.legend(loc=0)  # 指定legend的位置右下角
    return 1


# In[17]:

def judge(alist):
    if any(alist[i+1] < alist[i] for i in range(0, len(alist)-1)):
        return 0
    else:
        return 1


def getFitCCC(RLCQarray, fit_NRLCQ, selectIDX, i, allCCCtArray):
    MIN = min(RLCQarray[(selectIDX-i+1):selectIDX+1])
    ENarray = np.exp(1-(fit_NRLCQ*(1-MIN))-MIN)
    CCCt1 = allCCCtArray[0]
    fitCCClist = []
    for i in range(len(ENarray)):
        CCCt1 = ENarray[i]*CCCt1
        fitCCClist.append(CCCt1)
    return fitCCClist


def predictAccurateCCC(pred_len, selectcountid, selectdateidx, bicrange):
    datelist, RLCQarray, ccclist = getDateAndRLCQ(selectcountid)
    selectIDX = selectdateidx - 1
    iList = []
    bicList = []
    for i in range(5, bicrange):
        bic, NRLCQarray, fit_NRLCQ, pred_NRLCQ, allNRLCQarray, datelist, pred_datearray = fitNRLCQ(RLCQarray, selectIDX, i, pred_len)
        iList.append(i)
        bicList.append(bic)
    bicSelectIDX = iList[bicList.index(min(bicList))]
    bic, NRLCQarray, fit_NRLCQ, pred_NRLCQ, allNRLCQarray, datelist, pred_datearray = fitNRLCQ(RLCQarray, selectIDX, bicSelectIDX, pred_len)
    allCCCtArray = np.array(ccclist[(selectIDX-bicSelectIDX+1):selectIDX+pred_len+1])
    fitCCClist = getFitCCC(RLCQarray, fit_NRLCQ, selectIDX, bicSelectIDX, np.array(ccclist[(selectIDX-bicSelectIDX):selectIDX+pred_len+1]))  # 这个地方是获取拟合的值所转化出来的真实值
    r, p = stats.pearsonr(allCCCtArray[:len(allCCCtArray) - pred_len], fitCCClist)
    if judge(allCCCtArray) == 0 or p > 0.05:
        return 0, 1, -1
    predictCCClist = getPredictCCC(RLCQarray, NRLCQarray, pred_NRLCQ, selectIDX, bicSelectIDX, pred_len, allCCCtArray)
    pred_inc = predictCCClist[len(predictCCClist) - 1] - allCCCtArray[len(allCCCtArray) - len(predictCCClist) - 1]
    real_inc = allCCCtArray[len(allCCCtArray) - 1] - allCCCtArray[len(allCCCtArray) - len(predictCCClist) - 1]
    # print(abs((real_inc-pred_inc)/real_inc))
    return 1, abs((real_inc-pred_inc)/real_inc), bicSelectIDX

