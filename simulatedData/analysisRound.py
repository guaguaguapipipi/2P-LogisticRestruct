import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

dfori = pd.read_csv('virtualData.csv')
oridflist = dfori.values

beginday = 100

def fit_L3func(X, c1, c2, Cmax):
    return Cmax * c1 * np.exp(c2 * X) / (Cmax + c1 * (np.exp(c2 * X) - 1))

def fit_L2func(t, c1, c2):
    return 1 / (1 + (c1 * np.exp((-c2) * t)))

def getL2delta(dateArr, vOriArr):
    vOriArr = [float(i) for i in vOriArr]
    RLCArr = 1-(np.log(vOriArr[1:])-np.log(vOriArr[:-1]))
    minR = min(RLCArr)
    NRLCQ = (RLCArr - minR) / (1 - minR)
    dateArr = dateArr[1:]
    poptlist = []
    for i in range(len(dateArr)-beginday):
        popt, pcov = curve_fit(fit_L2func, dateArr[:(beginday+i)], NRLCQ[:(beginday+i)], p0=np.array([10000,0.01]),maxfev=20000000)
        poptlist.append(popt)
    return poptlist


def getL3delta(dateArr, vOriArr):
    poptlist = []
    for i in range(len(dateArr) - beginday):
        popt, pcov = curve_fit(fit_L3func, dateArr[:(beginday+i)], vOriArr[:(beginday+i)], p0=np.array([1,0,1]),maxfev=20000000)
        poptlist.append(popt)
    return poptlist

allres = []
for orilist in oridflist:
    oriarr = np.array(orilist)
    infostr = oriarr[0]
    l3str = infostr.split('&')[1]
    l3infolist = l3str.split(',')
    l3infolist = [info.split('=')[1] for info in l3infolist]
    noiselevel = l3infolist[3]
    l3infolist = l3infolist[:-1]
    l3infolist = [float(i) for i in l3infolist]
    print(l3infolist, end=' ')
    l2str = infostr.split('&')[0]
    l2infolist = [float(info) for info in l2str.split('=')[1].split('^^^')]
    print(l2infolist)
    oriarr = np.array(oriarr[1:])
    dateArr = np.array([i for i in range(len(oriarr))])


    l3poplist = getL3delta(dateArr, oriarr)
    l2poplist = getL2delta(dateArr, oriarr)
    allres = allres + l3poplist + l2poplist

print(allres)
resdf = pd.DataFrame(allres)
resdf.to_csv("allres.csv")