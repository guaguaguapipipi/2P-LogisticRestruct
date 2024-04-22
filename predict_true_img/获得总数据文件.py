import numpy as np
import pandas as pd

def read_one_csv(realfilepath, fitfilepath):
    realoridata = pd.read_csv(realfilepath, header=None).values
    fitoridata = pd.read_csv(fitfilepath, header=None).values
    totdata = []
    for i in range(len(realoridata)):
        totdata.append(list(realoridata[i]) + list(fitoridata[i][3:]))
    return totdata



def getallcsv(lens):
    countlen = 10
    datevalue = list(pd.read_csv("真实预测合并/51419-global_data.csv", header=None).values[0][102:448])
    l2 = pd.read_csv("真实预测合并/"+lens+"-L2.csv", index_col=0).values
    l3 = pd.read_csv("真实预测合并/"+lens + "-L3.csv", index_col=0).values
    lanl = pd.read_csv("真实预测合并/"+lens + "-LANL.csv", index_col=0).values
    sjka = pd.read_csv("真实预测合并/"+lens + "-SJKA.csv", index_col=0).values
    strlen = lens
    lens = int(strlen)
    alldatamat = []
    for i in l2:
        if i[2] == 0:
            onedatelist = datevalue[datevalue.index(i[1]):datevalue.index(i[1])+lens]
            for dateindex in range(len(onedatelist)):
                alldatamat.append([i[0]] + [onedatelist[dateindex]] + ["l2", dateindex, i[(3+lens)+dateindex]])
    for i in l3:
        if i[2] == 0:
            onedatelist = datevalue[datevalue.index(i[1]):datevalue.index(i[1])+lens]
            for dateindex in range(len(onedatelist)):
                alldatamat.append([i[0]] + [onedatelist[dateindex]] + ["l3", dateindex, i[(3+lens)+dateindex]])
    for i in lanl:
        if i[2] == 0:
            onedatelist = datevalue[datevalue.index(i[1]):datevalue.index(i[1])+lens]
            for dateindex in range(len(onedatelist)):
                alldatamat.append([i[0]] + [onedatelist[dateindex]] + ["lanl", dateindex, i[(3+lens)+dateindex]])
    for i in sjka:
        if i[2] == 0:
            onedatelist = datevalue[datevalue.index(i[1]):datevalue.index(i[1])+lens]
            for dateindex in range(len(onedatelist)):
                alldatamat.append([i[0]] + [onedatelist[dateindex]] + ["sjka", dateindex, i[(3+lens)+dateindex]])
    pd.DataFrame(alldatamat).to_csv("真实预测合并/"+strlen+"作图排版数据.csv")
    return


getallcsv("30")