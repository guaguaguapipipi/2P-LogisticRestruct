import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

dfori = pd.read_csv('virtualData.csv')
oridflist = dfori.values


def fit_L3func(X, c1, c2, Cmax):
    return Cmax * c1 * np.exp(c2 * X) / (Cmax + c1 * (np.exp(c2 * X) - 1))

def fit_L2func(t, c1, c2):
    return 1 / (1 + (c1 * np.exp((-c2) * t)))

def getL2Original(dateArr, vOriArr):
    vOriArr = [float(i) for i in vOriArr]
    RLCArr = 1-(np.log(vOriArr[1:])-np.log(vOriArr[:-1]))
    minR = 0.9
    NRLCQ = (RLCArr - minR) / (1 - minR)
    popt, pcov = curve_fit(fit_L2func, dateArr[1:], NRLCQ,
                           p0=np.array([1000, 0.1]), maxfev=20000000)
    return popt


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
    print(l3infolist, end='                                                    ')
    allres.append(l3infolist)
    l2str = infostr.split('&')[0]
    l2infolist = [float(info) for info in l2str.split('=')[1].split('^^^')]
    allres.append(l2infolist)
    print(l2infolist)
    oriarr = np.array(oriarr[1:])
    dateArr = np.array([i for i in range(len(oriarr))])
    popt, pcov = curve_fit(fit_L3func, dateArr, oriarr,
                           p0=np.array([1,1,1]), maxfev=20000000)
    print(popt, end=' ')
    deltal3 = 0
    deltal3list = []
    for i in range(len(popt)):
        deltal3 += (abs(l3infolist[i]-popt[i])/abs(l3infolist[i]))
        deltal3list.append((abs(l3infolist[i]-popt[i])/abs(l3infolist[i])))
    avgdelta = deltal3/3
    print('---',deltal3list,'---',avgdelta,'---',end='')
    allres.append(popt)
    popt = getL2Original(dateArr, oriarr)
    print(popt,end='')
    deltal2 = 0
    deltal2list = []
    for i in range(len(popt)):
        deltal2 += (abs(l2infolist[i] - popt[i]) / abs(l2infolist[i]))
        deltal2list.append((abs(l2infolist[i] - popt[i]) / abs(l2infolist[i])))
    avgdelta = deltal2 / 2
    print('---',deltal2list,'---',avgdelta)
    allres.append(popt)
resdf = pd.DataFrame(allres)
resdf.to_csv("allres.csv")