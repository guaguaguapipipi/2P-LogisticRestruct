import csv
import math
import time

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score




class L3PredCCCData:
    real_samp_ccc: np.array
    real_pred_ccc: np.array
    fit_samp_ccc: np.array
    fit_pred_ccc: np.array
    xic: float

    def __init__(self, xic, real_samp_ccc, real_pred_ccc, fit_samp_ccc, fit_pred_ccc):
        self.xic = xic
        self.real_samp_ccc = real_samp_ccc
        self.real_pred_ccc = real_pred_ccc
        self.fit_samp_ccc = fit_samp_ccc
        self.fit_pred_ccc = fit_pred_ccc
        return


class Model:
    _datearr = np.array([])  # 这里面保存的是所有日期
    _countryarr = np.array([])   # 这里面保存的是所有国家
    _Carr = np.array([])   # 这里面保存的是所有国家的确诊数据
    _pred_len = 7  # 这个是要预测的长度

    _predictCountIDX = 0
    _predictIDXList = []  # 这个是已经预测了序列的

    _allCountXicList = []

    _pLimit = 100

    _XICFrequencyOfDistribution = []  # 这个是xic选择的值的频数分布表

    _originalWight = 0.5
    _maxWight = 0.5
    _dramaticRange = 0

    _csvResultsName = ''

    _dramaticList = []

    _windowSizeBegin = 5  # 窗口大小的开始
    _windowSizeEnd = 20  # 窗口大小最大

    def setWindowSizeBegin(self, windowsizeBeg):
        self._windowSizeBegin = windowsizeBeg

    def setWindowSizeEnd(self, windowsizeEnd):
        self._windowSizeEnd = windowsizeEnd

    def setCsvResultsName(self, filename):
        self._csvResultsName = filename

    def setWight(self, w:float):
        self._originalWight = w
        return

    def setWightMax(self, w:float):
        self._maxWight = w
        return

    def setDramaticRange(self, d:float):
        self._dramaticRange = d
        return

    def setPlimit(self, pl):
        self._pLimit = pl
        return

    def getAllCountXicList(self):
        return self._allCountXicList

    def getDramaticList(self):
        return self._dramaticList

    def setCountIDX(self, countIDX):
        self._predictCountIDX = countIDX
        return

    def setPredlen(self, pred_len):
        self._pred_len = pred_len
        return

    def BIC(self, Y, pred_Y, k):
        # pred_Y是预测值，Y是实际值，k是参数的数量
        Q = np.sum((np.array(Y) - np.array(pred_Y)) ** 2)
        n = len(Y)
        lnL = -n / 2 * (np.log(2 * np.pi * Q / n) + 1)
        bic = -2 * lnL + np.log(n) * k
        return bic

    def AIC(self, Y, pred_Y, k):
        # pred_Y是预测值，Y是实际值，k是参数的数量
        Q = np.sum((np.array(Y)-np.array(pred_Y))**2)
        n = len(Y)
        lnL = -n/2*(np.log(2*np.pi*Q/n)+1)
        bic = 2*k-2*lnL
        return bic

    def __init__(self, filename, predlen):
        self._pred_len = predlen
        while 1:
            try:
                datadict = {}
                with open(filename, encoding='utf-8-sig') as csvfile:
                    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                    for row in csv_reader:  # 将csv 文件中的数据保存到data中
                        datadict[row[0]] = row[1:]
                    csvfile.close()
                break
            except Exception as e:
                continue
        self._datearr = np.array(datadict["Country/Region"])
        self._countryarr = np.array(list(datadict.keys())[1:])
        clist = []
        idx = 0
        # 这里是初始化确证人数的数组
        for count in self._countryarr:
            clist.insert(idx, datadict[count])
            idx += 1
        self._Carr = np.array(clist).astype(float)
        return

    def fit_L3func(self, X, c1, c2, Cmax):
        return Cmax * c1 * np.exp(c2 * X) / (Cmax + c1 * (np.exp(c2 * X) - 1))

    def predict_CCC(self, selectIDX, bicIDX):
        self._predictIDXList.append(selectIDX)
        CCC_real_samp = self._Carr[self._predictCountIDX][selectIDX-bicIDX:selectIDX]
        CCC_real_pred = self._Carr[self._predictCountIDX][selectIDX:selectIDX+self._pred_len]
        samp_date = np.array([j for j in range(len(CCC_real_samp))])
        pred_date = np.array([j for j in range(len(CCC_real_samp), len(CCC_real_samp)+self._pred_len)])
        popt, pcov = curve_fit(self.fit_L3func, samp_date, CCC_real_samp, p0=np.array([CCC_real_samp[0], 1, CCC_real_samp[-1]]),maxfev=2000000000)
        CCC_fit_samp = self.fit_L3func(samp_date, popt[0], popt[1], popt[2])
        CCC_fit_pred = self.fit_L3func(pred_date, popt[0], popt[1], popt[2])
        xic = self.AIC(CCC_real_pred, CCC_fit_pred, 3)
        L3Data = L3PredCCCData(xic, CCC_real_samp, CCC_real_pred, CCC_fit_samp, CCC_fit_pred)
        return L3Data

    def isDramatic2(self, cccList):
        Dvalue = cccList[1:] - cccList[:-1]
        g3 = sum(Dvalue[0:3])/3
        g2 = sum(Dvalue[3:5])/2
        g4 = sum(Dvalue[0:4])/4
        g1 = Dvalue[4]
        d1 = abs((g2-g3)/g3)
        d2 = abs((g1-g4)/g4)
        if d1 >= self._dramaticRange or d2 >= self._dramaticRange:
            self.setWight(0)
        else:
            self.setWight(self._maxWight)
        # self._dramaticList.append([d1,d2])
        return

    def getRankList(self, xicList, lastApeList):
        aicRankList = [0 for i in range(len(xicList))]
        apeRankList = [0 for i in range(len(lastApeList))]
        sortList = [key for key ,value in sorted(enumerate(xicList), key=itemgetter(1))]
        for i in range(len(xicList)):
            aicRankList[sortList[i]] = i + 1
        sortList = [key for key, value in sorted(enumerate(lastApeList), key=itemgetter(1))]
        for i in range(len(lastApeList)):
            apeRankList[sortList[i]] = i + 1
        scoreList = (np.array(aicRankList) * self._originalWight + np.array(apeRankList) * (
                    1 - self._originalWight)).tolist()
        return scoreList

    def judge(self, alist):
        if any(alist[i + 1] < alist[i] for i in range(0, len(alist) - 1)):
            return 0
        else:
            return 1

    def predictRightCCC(self, selectcountid, selectdateidx):
        iList = []
        xicList = []
        lastApeList = []
        iserror = 0
        self._predictCountIDX = selectcountid
        self.isDramatic2(self._Carr[selectcountid][selectdateidx - self._windowSizeBegin - 1:selectdateidx])
        for i in range(self._windowSizeBegin, self._windowSizeEnd + 1):
            try:
                L3Data = self.predict_CCC(selectdateidx, i)
            except:
                return 1, 0, 0, 0, 0
            iList.append(i)
            xicList.append(L3Data.xic)
            lastApeList.append(abs(L3Data.real_samp_ccc[-1] - L3Data.fit_samp_ccc[-1]))
        rankList = self.getRankList(xicList, lastApeList)
        xicList = rankList
        xicSelectIDX = iList[xicList.index(min(xicList))] # 选择一个权重排名最小的
        L3Data = self.predict_CCC(selectdateidx, xicSelectIDX)
        real_samp_ccc = L3Data.real_samp_ccc
        real_pred_ccc = L3Data.real_pred_ccc
        fit_samp_ccc = L3Data.fit_samp_ccc
        fit_pred_ccc = L3Data.fit_pred_ccc
        pred_inc = fit_pred_ccc[len(fit_pred_ccc) - 1] - real_samp_ccc[len(real_samp_ccc) - 1]
        real_inc = real_pred_ccc[len(real_pred_ccc) - 1] - real_samp_ccc[len(real_samp_ccc) - 1]
        ape = abs((real_inc - pred_inc) / real_inc)
        # print(ape)
        # r, p = stats.pearsonr(real_samp_ccc, fit_samp_ccc)
        if self.judge(list(real_samp_ccc) + list(real_pred_ccc)) == 0:
            iserror = 1
        return iserror, L3Data, xicSelectIDX, ape, xicList

    def predictOneCountry(self, selectcountid):
        print("国家序号为：", end='')
        print(selectcountid)
        resultsList = []
        xicSelectIdxList = []
        countryXicList = []
        for i in range(self._windowSizeEnd+1):
            xicSelectIdxList.append(0)
        firstdate = "30"
        lastdate = "280"
        idxfirstdate = list(self._datearr).index(firstdate)
        idxlastdate = list(self._datearr).index(lastdate)
        print(str(idxfirstdate) + " : " + self._datearr[idxfirstdate])
        print(str(idxlastdate) + " : " + self._datearr[idxlastdate])
        for dateidx in range(idxfirstdate, idxlastdate+1):
            iserror, L3Data, xicSelectIDX, ape, xicList = self.predictRightCCC(selectcountid, dateidx)
            xicSelectIdxList[xicSelectIDX] += 1
            if iserror == 1 or math.isnan(ape) or math.isinf(ape):
                continue
            resultsList.append(ape)
            countryXicList.append(xicList)
        resultsArr = np.array(resultsList)
        plt.figure()
        plt.subplot(1, 1, 1)
        mapDateList = [i for i in range(len(resultsList))]
        plt.plot(mapDateList, resultsList, 'r', color="blue")
        plt.show()
        print("这个是L3的推了{}次的平均误差和标准差：{}±{}，最大值为：{}".format(
            resultsArr.size, np.mean(resultsArr), np.std(resultsArr), max(resultsArr)))
        print(xicSelectIdxList)
        with open(self._csvResultsName, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            data_row = [selectcountid, '', idxfirstdate, idxlastdate, firstdate,
                        lastdate, np.mean(resultsArr), np.std(resultsArr),
                        str(np.mean(resultsArr)) + "&&&" + str(np.std(resultsArr)), resultsArr.size, max(resultsArr)]
            csv_write.writerow(data_row)
        return countryXicList

    def predictListCountry(self, countList):
        with open(self._csvResultsName, 'a+') as f:
            csv_write = csv.writer(f)
            csv_write.writerow([])
            csv_write.writerow(["WIGHT&THRESHOLD&" + str(self._maxWight) + "&" + str(self._dramaticRange)])
            f.close
        for count in countList:
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            countryXicList = self.predictOneCountry(count)
            self._allCountXicList.append(countryXicList)
        return

def traversalAllRange():
    startTime = time.time()
    model = Model("这个是生成的虚拟数据.csv", 7)
    for i in range(2,9):
        for j in range(2, 9):
            model.setPlimit(0.01)
            model.setWightMax(i/10)
            model.setDramaticRange(j/10)
            model.setCsvResultsName("这个是生成的虚拟数据L3跑的结果.csv")
            # countlist = [155, 21, 71, 98, 153, 77, 56, 73, 138, 124]
            # countlist = [0]
            # countlist = [0, 56, 74, 80, 95, 128, 156, 162]  # 这个所有国家里面的
            countlist = [i for i in range(5)]
            print("----------------------------------------------------------------------------------------")
            model.setWindowSizeBegin(14)
            model.setWindowSizeEnd(21)
            model.predictListCountry(countlist)
            allCountXic = model.getAllCountXicList()
    endTime = time.time()
    print(startTime-endTime,'s')

def traversalOneRange(wight, dramatic):
    startTime = time.time()
    model = Model("这个是生成的虚拟数据.csv", 7)
    model.setPlimit(0.01)
    model.setWightMax(wight)
    model.setDramaticRange(dramatic)
    model.setCsvResultsName("这个是生成的虚拟数据L3跑的结果.csv")
    countlist = [i for i in range(5)]
    print("----------------------------------------------------------------------------------------")
    model.setWindowSizeBegin(14)
    model.setWindowSizeEnd(21)
    model.predictListCountry(countlist)
    allCountXic = model.getAllCountXicList()
    endTime = time.time()
    print(startTime - endTime, 's')

traversalAllRange()
# traversalOneRange(0.3, 0.4)