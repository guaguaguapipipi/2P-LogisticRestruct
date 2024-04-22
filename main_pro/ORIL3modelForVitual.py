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

    def predictRightCCC(self, selectcountid, dateidx):
        real_samp_ccc = self._Carr[selectcountid][:dateidx]
        real_pred_ccc = self._Carr[selectcountid][dateidx:dateidx+self._pred_len]
        samp_date = np.array([i for i in range(len(real_samp_ccc))])
        pred_date = np.array([i for i in range(len(real_samp_ccc), len(real_samp_ccc)+self._pred_len)])
        popt, pcov = curve_fit(self.fit_L3func, samp_date, real_samp_ccc, p0=np.array([real_samp_ccc[0], 1, real_samp_ccc[-1]]),maxfev=2000000000)
        fit_samp_ccc = self.fit_L3func(samp_date, popt[0], popt[1], popt[2])
        fit_pred_ccc = self.fit_L3func(pred_date, popt[0], popt[1], popt[2])
        pred_inc = fit_pred_ccc[len(fit_pred_ccc) - 1] - real_samp_ccc[len(real_samp_ccc) - 1]
        real_inc = real_pred_ccc[len(real_pred_ccc) - 1] - real_samp_ccc[len(real_samp_ccc) - 1]
        ape = abs((real_inc - pred_inc) / real_inc)
        L3Data = L3PredCCCData(real_samp_ccc, real_pred_ccc, fit_samp_ccc, fit_pred_ccc)
        return L3Data, ape


    def predictOneCountry(self, selectcountid):
        print("国家序号为：", end='')
        print(selectcountid)
        resultsList = []
        firstdate = "30"
        lastdate = "280"
        idxfirstdate = list(self._datearr).index(firstdate)
        idxlastdate = list(self._datearr).index(lastdate)
        print(str(idxfirstdate) + " : " + self._datearr[idxfirstdate])
        print(str(idxlastdate) + " : " + self._datearr[idxlastdate])
        for dateidx in range(idxfirstdate, idxlastdate + 1):
            L3Data, ape = self.predictRightCCC(selectcountid, dateidx)
            resultsList.append(ape)
        resultsArr = np.array(resultsList)
        print("这个是L3的推了{}次的平均误差和标准差：{}±{}，最大值为：{}".format(
            resultsArr.size, np.mean(resultsArr), np.std(resultsArr), max(resultsArr)))
        with open(self._csvResultsName, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            data_row = [selectcountid, '', idxfirstdate, idxlastdate, firstdate,
                        lastdate, np.mean(resultsArr), np.std(resultsArr),
                        str(np.mean(resultsArr)) + "&&&" + str(np.std(resultsArr)), resultsArr.size, max(resultsArr)]
            csv_write.writerow(data_row)
        return

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


def traversalOneRange():
    startTime = time.time()
    model = Model("这个是生成的虚拟数据.csv", 7)
    model.setCsvResultsName("这个是纯L3生成的虚拟数据的运行数据.csv")
    countlist = [i for i in range(8)]
    print("----------------------------------------------------------------------------------------")
    model.predictListCountry(countlist)
    endTime = time.time()
    print(startTime - endTime, 's')


traversalOneRange()