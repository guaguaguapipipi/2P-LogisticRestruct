import csv
import math
import time

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class Model:
    _datearr = np.array([])  # 这里面保存的是所有日期
    _countryarr = np.array([])  # 这里面保存的是所有国家
    _Carr = np.array([])  # 这里面保存的是所有国家的确诊数据
    _RLCQarrr = np.array([])  # 这里面保存的是所有国家的RLCCCQ
    _pred_len = 7  # 这个是要预测的长度
    _RLCQMvList = []  # 这个是放RLCQ的偏移量（因为log不能对0使用）

    _predictCountIDX = 0
    _predictIDXList = []  # 这个是已经预测了序列的

    _allCountXicList = []

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
                print(1)
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
        print(self._RLCQarrr[0])
        return


model2 = Model("这个是生成的虚拟数据.csv", 7)