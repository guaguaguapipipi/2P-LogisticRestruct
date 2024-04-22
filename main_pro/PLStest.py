import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression


def XY_get(windowsize):
    x = np.array([i for i in range(1, windowsize+1)])
    xmat = []
    for i in range(1,windowsize+1):
        xmat.append(np.abs(i-x))
    ymat = []
    ymat = [random.random() for i in range(1, windowsize+1)]
    xmat = np.array(xmat)
    ymat = np.array(ymat)
    return xmat, ymat

minwindow = 5

maxwindow = 30
meanScoreList = []
for j in range(minwindow, maxwindow+1):
    oneScoreList = []
    for i in range(10000):
        xmat, ymat = XY_get(j)
        model = PLSRegression(n_components=3)
        model.fit(xmat, ymat)
        oneScoreList.append(model.score(xmat, ymat))
    oneArray = np.array(oneScoreList)
    meanScoreList.append(oneArray.mean())
    print(meanScoreList)
timeList = [i for i in range(minwindow, maxwindow+1)]

plt.plot(timeList, meanScoreList, 'r', label='mean')
plt.legend(loc='best')
plt.show()