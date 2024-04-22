import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
import palettable
from sklearn import datasets
from matplotlib import ticker
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import MaxNLocator

countlistname = ["US","Brazil","India","Mexico","UK","Italy","France","Iran","Spain","Russia"]

def getonecountplot(countidx, strlen, partidx):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文
    oridata = pd.read_csv(strlen+"作图排版数据.csv", index_col=0)
    realdata = pd.read_csv("51419-global_data.csv", header=None).T
    realdata.drop([0], inplace=True)
    realdata[0] = pd.to_datetime(realdata[0])
    oridata['date'] = pd.to_datetime(oridata['date'])
    oridata = oridata.sort_values(by=["date"])

    count1data = oridata[oridata['country'].isin([countidx])]
    # 切片
    partlen = int(len(count1data) / 20)
    count1data = count1data.iloc[partlen*(partidx-1):partlen*partidx]
    # 切片
    firstday = (count1data.iloc[0])['date']
    lastday = (count1data.iloc[len(count1data)-1])['date']
    plotdata = realdata[(firstday<=realdata[0]) & (lastday>=realdata[0])]
    plt.figure(dpi=1000)
    datedict = dict({})
    inc = 1
    for i in plotdata[0]:
        datedict[i] = inc
        inc += 1

    for i in range(len(count1data)):
        count1data.iloc[i, 1] = int(datedict[count1data.iloc[i, 1]])
        if count1data.iloc[i, 2] == 'l3':
            count1data.iloc[i, 2] = 'LGM'
        if count1data.iloc[i, 2] == 'l2':
            count1data.iloc[i, 2] = 'L2'
        if count1data.iloc[i, 2] == 'lanl':
            count1data.iloc[i, 2] = 'LANL'
        if count1data.iloc[i, 2] == 'sjka':
            count1data.iloc[i, 2] = 'SIKJα'

    for i in range(len(plotdata)):
        plotdata.iloc[i, 0] = int(datedict[plotdata.iloc[i, 0]])
        plotdata.iloc[i, 1] = np.float64(plotdata.iloc[i, 1])
        plotdata.iloc[i, 2] = np.float64(plotdata.iloc[i, 2])
        plotdata.iloc[i, 3] = np.float64(plotdata.iloc[i, 3])
        plotdata.iloc[i, 4] = np.float64(plotdata.iloc[i, 4])
        plotdata.iloc[i, 5] = np.float64(plotdata.iloc[i, 5])
        plotdata.iloc[i, 6] = np.float64(plotdata.iloc[i, 6])
        plotdata.iloc[i, 7] = np.float64(plotdata.iloc[i, 7])
        plotdata.iloc[i, 8] = np.float64(plotdata.iloc[i, 8])
        plotdata.iloc[i, 9] = np.float64(plotdata.iloc[i, 9])
        plotdata.iloc[i, 10] = np.float64(plotdata.iloc[i, 10])
        # print(type(plotdata.iloc[i, 1]))
        # print(type(plotdata.iloc[i, 0]))

    fig, ax1 = plt.subplots()

    sns.boxplot(x=list(count1data["date"].T.values),
                y=count1data["Cumulative Confirmed Cases"],
                hue=count1data["model"],
                orient='v',
                linewidth=1.2,
                fliersize=0,
                # boxprops = {
                #     'color':'white'
                # },
                hue_order=['L2','LGM','SIKJα','LANL'],
                # palette=palettable.colorbrewer.qualitative.Set3_4.mpl_colors,
                palette=plt.get_cmap('Accent')(range(1,5)),
                width=1,
                )
    font3 = {'family': 'Times New Roman',
             'weight': 'bold',}
    ax1.get_legend().remove()
    ax1.legend(loc="lower right",prop=font3)


    #ax2 = ax1.twinx()
    plotdata.columns = ["date", '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    # plotdata.iloc[1].astype(np.float64)
    realdatalist = plotdata.T.values
    # sns.lineplot(x='date', y='1', data=plotdata[['date', '1']])
    plt.plot(realdatalist[0], realdatalist[countidx+1], marker='o', markersize=2, linewidth=1, color='black')

    #ax2.set_ylim(1300000, 2300000)
    # ax1.set_ylim(min(realdatalist[countidx+1])*(1-0.1), max(realdatalist[countidx+1])*(1+0.1))
    miny = min(realdatalist[countidx + 1])
    maxy = max(realdatalist[countidx + 1])
    ax1.set_ylim(miny-(maxy-miny)*0.1, maxy+(maxy-miny)*0.1)



    # plt.xlabel(countlistname[countidx],fontsize=20)
    # plt.xlabel("Date", fontsize=20)
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 20,
             }
    plt.xlabel("Date", font1)
    font2 = {'family': 'Times New Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 20,
             }
    plt.ylabel("Cumulative Confirmed Cases", font2)
    # plt.ylabel("Cumulative Confirmed Cases", fontsize=20)
    plt.tick_params(labelsize=16)

    x_major_locator = MultipleLocator(3)  # 以每15显示
    ax1.xaxis.set_major_locator(x_major_locator)

    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.FormatStrFormatter('%.2f')
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((0, 4))
    formatter = ticker.FuncFormatter(formatnum)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_locator(MaxNLocator(2))
    # for i, tick in enumerate(ax1.yaxis.get_ticklabels()):
    #     if i % 2 != 0:
    #         tick.set_visible(False)
    plt.savefig('img4/' + strlen + '-' + str(partidx) + '-' + str(countidx) + '.png', bbox_inches='tight', dpi=1000)
    # plt.show()

def formatnum(x, pos):
    # 表示保留3位小数点，-2为e的次方数，x的值乘以10的2次方就可以转换为科学计数法的值
    return '$%.2f$' % (x/10000000)

# for countid in range(10):
#     for partid in range(1,21):
#         getonecountplot(countid, '07', partid)
#         # getonecountplot(countid, '15', partid)
#         # getonecountplot(countid, '30', partid)
getonecountplot(0, '15', 17)
getonecountplot(1, '15', 17)
getonecountplot(2, '15', 17)
getonecountplot(3, '15', 17)
getonecountplot(4, '15', 17)
getonecountplot(9, '15', 17)
# getonecountplot(0, '15', 8)
# getonecountplot(1, '15', 8)
# getonecountplot(2, '15', 8)
# getonecountplot(3, '15', 8)
# getonecountplot(4, '15', 8)
# getonecountplot(9, '15', 8)