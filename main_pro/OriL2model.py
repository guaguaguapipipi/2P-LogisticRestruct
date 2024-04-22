import csv
import math
import time

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


class L2PredData:
    xic: float
    NRLCQ_real_samp: np.array
    NRLCQ_fit_samp: np.array
    NRLCQ_real_pred: np.array
    NRLCQ_fit_pred: np.array
    samp_date: np.array
    pred_date: np.array
    RLCQ_real_samp: np.array
    RLCQ_real_pred: np.array
    def __init__(self, xic, NRLCQ_real_samp, NRLCQ_fit_samp, NRLCQ_real_pred,
                 NRLCQ_fit_pred, samp_date, pred_date, RLCQ_real_samp, RLCQ_real_pred):
        self.xic = xic
        self.NRLCQ_real_samp = NRLCQ_real_samp
        self.NRLCQ_fit_samp = NRLCQ_fit_samp
        self.NRLCQ_real_pred = NRLCQ_real_pred
        self.NRLCQ_fit_pred = NRLCQ_fit_pred
        self.samp_date = samp_date
        self.pred_date = pred_date
        self.RLCQ_real_samp = RLCQ_real_samp
        self.RLCQ_real_pred = RLCQ_real_pred
        return


class L2PredCCCData:
    real_samp_ccc: np.array
    real_pred_ccc: np.array
    fit_samp_ccc: np.array
    fit_pred_ccc: np.array

    def __init__(self, real_samp_ccc, real_pred_ccc, fit_samp_ccc, fit_pred_ccc):
        self.real_samp_ccc = real_samp_ccc
        self.real_pred_ccc = real_pred_ccc
        self.fit_samp_ccc = fit_samp_ccc
        self.fit_pred_ccc = fit_pred_ccc
        return


class Model:
    _datearr = np.array([])  # 这里面保存的是所有日期
    _countryarr = np.array([])   # 这里面保存的是所有国家
    _Carr = np.array([])   # 这里面保存的是所有国家的确诊数据
    _RLCQarrr = np.array([])   # 这里面保存的是所有国家的RLCCCQ
    _pred_len = 7  # 这个是要预测的长度
    _RLCQMvList = []  # 这个是放RLCQ的偏移量（因为log不能对0使用）

    _predictCountIDX = 0
    _predictIDXList = []  # 这个是已经预测了序列的

    _allCountXicList = []

    _pLimit = 100

    _XICFrequencyOfDistribution = []  # 这个是xic选择的值的频数分布表

    _originalWight = 1
    _maxWight = 1
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
        # 这里是初始化RLCCCQ的数组,对齐了的
        rlcqlist = []
        idx = 0
        for i in range(len(self._countryarr)):
            try:
                rlcqlist.insert(idx, np.insert(1 - (np.log(self._Carr[i][1:]) - np.log(self._Carr[i][:-1])), 0, np.nan))
            except Exception as e:
                print(self._Carr[i])
                rlcqlist.insert(idx, [0])
            idx += 1
        self._RLCQarrr = np.array(rlcqlist)
        for i in rlcqlist:
            for d in range(len(i)):
                if i[d] == float("-inf") or np.isnan(i[d]):
                    continue
                else:
                    self._RLCQMvList.append(d)
                    break
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

    def AICwithoutN(self, Y, pred_Y, k):
        # pred_Y是预测值，Y是实际值，k是参数的数量
        Q = np.sum((np.array(Y)-np.array(pred_Y))**2)
        n = len(Y)
        lnL = -n/2*(np.log(2*np.pi*Q/n)+1)
        bic = 2*k-2*lnL
        return bic

    def fit_L2func(self, t, c1, c2):
        return 1 / (1 + (c1 * np.exp((-c2) * t)))

    def analysisNRLCQ(self, RLCQarray, MIN):
        """
            如果是预测的时候调用,那MIN要设定为None,如果是获取实际值的时候就要在MIN中放入前段参考值中的最小值
        """
        if MIN is None:
            MIN = min(RLCQarray)
        NRLCQarray = (RLCQarray - MIN) / (1 - MIN)
        return NRLCQarray

    def predict_NRLCQ(self, selectIDX, bicIDX):
        self._predictIDXList.append(selectIDX)
        RLCQ_real_samp = self._RLCQarrr[self._predictCountIDX][selectIDX-bicIDX:selectIDX]
        RLCQ_real_pred = self._RLCQarrr[self._predictCountIDX][selectIDX:selectIDX+self._pred_len]
        NRLCQ_real_samp = self.analysisNRLCQ(RLCQ_real_samp, None)
        NRLCQ_real_pred = self.analysisNRLCQ(RLCQ_real_pred, min(RLCQ_real_samp))
        samp_date = np.array([j for j in range(len(NRLCQ_real_samp))])
        pred_date = np.array([j for j in range(len(NRLCQ_real_samp), len(NRLCQ_real_samp)+self._pred_len)])
        popt, pcov = curve_fit(self.fit_L2func, samp_date, NRLCQ_real_samp, maxfev=20000000)
        NRLCQ_fit_samp = self.fit_L2func(samp_date, popt[0], popt[1])
        NRLCQ_fit_pred = self.fit_L2func(pred_date, popt[0], popt[1])
        # xic = self.AIC(NRLCQ_real_pred, NRLCQ_fit_pred, 2)
        xic = self.AIC(NRLCQ_real_samp, NRLCQ_fit_samp, 2)
        L2Data = L2PredData(xic, NRLCQ_real_samp, NRLCQ_fit_samp, NRLCQ_real_pred,
                            NRLCQ_fit_pred, samp_date, pred_date, RLCQ_real_samp, RLCQ_real_pred)
        return L2Data

    def getFitCCC(self, lastdateidx, fitNRLCQarr, MIN):
        beforeSelectDateCCC = self._Carr[self._predictCountIDX][lastdateidx-len(fitNRLCQarr)]
        ENarray = np.exp(1 - (fitNRLCQarr * (1 - MIN)) - MIN)
        ccct1 = beforeSelectDateCCC
        fitcccList = []
        for i in ENarray:
            ccct1 = i*ccct1
            fitcccList.append(ccct1)
        return np.array(fitcccList)

    def judge(self, alist):
        if any(alist[i + 1] < alist[i] for i in range(0, len(alist) - 1)):
            return 0
        else:
            return 1

    def getRankList(self, xicList, lastApeList):
        aicRankList = [0 for i in range(len(xicList))]
        apeRankList = [0 for i in range(len(lastApeList))]
        sortList = [key for key,value in sorted(enumerate(xicList), key=itemgetter(1))]
        for i in range(len(xicList)):
            aicRankList[sortList[i]] = i + 1
        sortList = [key for key,value in sorted(enumerate(lastApeList), key=itemgetter(1))]
        for i in range(len(lastApeList)):
            apeRankList[sortList[i]] = i + 1
        scoreList = (np.array(aicRankList)*self._originalWight + np.array(apeRankList)*(1-self._originalWight)).tolist()
        return scoreList

    def isDramatic(self, cccList):
        Dvalue = cccList[1:] - cccList[:-1]
        d1 = (sum(Dvalue[0:3])/3)/(sum(Dvalue[3:5])/2)
        d2 = (sum(Dvalue[0:4])/4)/(Dvalue[4])
        if abs(d1-d2) >= self._dramaticRange :
            self.setWight(0)
        else:
            self.setWight(self._maxWight)
        # self._dramaticList.append([d1,d2])
        return

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


    def predictAccurateCCC(self, selectcountid, selectdateidx):
        iList = []
        xicList = []
        lastApeList = []
        iserror = 0
        self._predictCountIDX = selectcountid
        # self.isDramatic2(self._Carr[selectcountid][selectdateidx - self._windowSizeBegin - 1:selectdateidx])
        for i in range(self._windowSizeBegin, self._windowSizeEnd+1):
            try:
                L2Data = self.predict_NRLCQ(selectdateidx, i)
            except:
                return 1, 0, 0, 0, 0, 0
            iList.append(i)
            xicList.append(L2Data.xic)
            real_samp_ccc = self._Carr[selectcountid][selectdateidx - i:selectdateidx]
            fit_samp_ccc = self.getFitCCC(selectdateidx - 1, L2Data.NRLCQ_fit_samp, min(L2Data.RLCQ_real_samp))
            # lastApeList.append(abs(real_samp_ccc[-1]-fit_samp_ccc[-1]))
        # rankList = self.getRankList(xicList, lastApeList)
        # xicList = rankList
        xicSelectIDX = iList[xicList.index(min(xicList))] # 选择一个权重排名最小的
        L2Data = self.predict_NRLCQ(selectdateidx, xicSelectIDX)
        real_samp_ccc = self._Carr[selectcountid][selectdateidx-xicSelectIDX:selectdateidx]
        real_pred_ccc = self._Carr[selectcountid][selectdateidx:selectdateidx+self._pred_len]
        fit_samp_ccc = self.getFitCCC(selectdateidx-1, L2Data.NRLCQ_fit_samp, min(L2Data.RLCQ_real_samp))
        fit_pred_ccc = self.getFitCCC(selectdateidx+self._pred_len-1, L2Data.NRLCQ_fit_pred, min(L2Data.RLCQ_real_samp))
        L2cccData = L2PredCCCData(real_samp_ccc, real_pred_ccc, fit_samp_ccc, fit_pred_ccc)
        pred_inc = fit_pred_ccc[len(fit_pred_ccc)-1] - real_samp_ccc[len(real_samp_ccc)-1]
        real_inc = real_pred_ccc[len(real_pred_ccc)-1] - real_samp_ccc[len(real_samp_ccc)-1]
        ape = abs((real_inc-pred_inc)/real_inc)
        # print(ape)
        r, p = stats.pearsonr(real_samp_ccc, fit_samp_ccc)
        if self.judge(list(real_samp_ccc)+list(real_pred_ccc)) == 0 or p >= self._pLimit:
            iserror = 1
        return iserror, L2Data, L2cccData, xicSelectIDX, ape, xicList

    def predictOneCountry(self, selectcountid):
        print("国家序号为：", end='')
        print(selectcountid)
        resultsList = []
        xicSelectIdxList = []
        countryXicList = []
        for i in range(self._windowSizeEnd+1):
            xicSelectIdxList.append(0)
        firstdate = "7/1/21"
        lastdate = "5/27/22"
        idxfirstdate = list(self._datearr).index(firstdate)
        idxlastdate = list(self._datearr).index(lastdate)
        print(str(idxfirstdate) + " : " + self._datearr[idxfirstdate])
        print(str(idxlastdate) + " : " + self._datearr[idxlastdate])
        for dateidx in range(idxfirstdate, idxlastdate+1):
            iserror, L2Data, L2cccData, xicSelectIDX, ape, xicList = self.predictAccurateCCC(selectcountid, dateidx)
            if iserror != 1 and ape >= 20:
                plt.figure()
                plt.subplot(1, 1, 1)
                plt.title("ErrorData", loc="right")
                plt.xlabel('date')
                plt.ylabel('CCC')
                mapDateList = [mapd for mapd in range(len(L2cccData.real_samp_ccc)+len(L2cccData.real_pred_ccc))]
                mapSampList = mapDateList[:-7]
                mapPredList = mapDateList[-7:]
                plt.plot(mapSampList, L2cccData.real_samp_ccc, 'r', color="blue", label="original")
                plt.plot(mapPredList, L2cccData.real_pred_ccc, 'r', color="blue", label="original")
                plt.plot(mapSampList, L2cccData.fit_samp_ccc, 'r', color="red", label="predict")
                plt.plot(mapPredList, L2cccData.fit_pred_ccc, 'r', color="red", label="predict")
                plt.show()
                print("序列中错误值出现位置为：",str(dateidx),end="序列为：")
                print(L2cccData.real_samp_ccc)
            xicSelectIdxList[xicSelectIDX] += 1
            if iserror == 1 or math.isnan(ape) or math.isinf(ape):
                continue
            resultsList.append(ape)
            countryXicList.append(xicList)
        resultsArr = np.array(resultsList)
        print("这个是双rank的推了{}次的平均误差和标准差：{}±{}，最大值为：{}".format(
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
    model = Model("所有国家的新确诊数据.csv", 7)
    for i in range(2,4):
        for j in range(2, 4):
            model.setPlimit(0.01)
            model.setWightMax(i/10)
            model.setDramaticRange(j/10)
            model.setCsvResultsName("所有国家的一部分遍历.csv")
            # countlist = [155, 21, 71, 98, 153, 77, 56, 73, 138, 124]
            # countlist = [0]
            countlist = [i for i in range(174)]
            print("----------------------------------------------------------------------------------------")
            model.setWindowSizeBegin(7)
            model.setWindowSizeEnd(21)
            model.predictListCountry(countlist)
            allCountXic = model.getAllCountXicList()
            # f = open('正确的死亡的遍历MaxMightRankData'+str(i/10)+'e'+str(j/10)+'.py', 'w')
            # print(allCountXic, file=f)
            # f.close()
    endTime = time.time()
    print(startTime-endTime,'s')

def traversalOneRange(wight, dramatic):
    startTime = time.time()
    model = Model("所有国家的新确诊数据.csv", 7)
    model.setPlimit(0.01)
    model.setWightMax(wight)
    model.setDramaticRange(dramatic)
    model.setCsvResultsName("去除掉权重判断只aic所有国家的一部分遍历.csv")
    # countlist = [155, 21, 71, 98, 153, 77, 56, 73, 138, 124]
    countlist = [i for i in range(174)]
    print("----------------------------------------------------------------------------------------")
    model.setWindowSizeBegin(14)
    model.setWindowSizeEnd(21)
    model.predictListCountry(countlist)
    allCountXic = model.getAllCountXicList()
    # f = open('正确的死亡的遍历MaxMightRankData' + str(wight) + 'e' + str(dramatic) + '.py', 'w')
    # print(allCountXic, file=f)
    # f.close()
    endTime = time.time()
    print(startTime - endTime, 's')

# traversalAllRange()
traversalOneRange(0.3, 0.4)