""" arima 预测各大洲的产量 """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 解决matlab画图中文乱码问题
from pylab import mpl
import statsmodels.api as sm

class ARIMA_predict():
   def run(self):
       mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
       data = pd.read_csv("csv/Asia_production_quantity.csv")
       # 将Years中的数据用时间表示出，为之后重采样等一系列操作做准备
       data['dataTime'] = pd.date_range('1961/01/01', periods=65, freq='Y')
       # 将添加的时间列设为索引
       data.set_index(['dataTime'], inplace=True)

       train = data[0:30]  # 训练数据
       test = data[30:]  # 测试数据

       # 原图
       plt.plot(data['Year'], data['Value'])
       plt.title('中国大豆产量')
       plt.show()

       def showplot(a, b, c, name):
           plt.figure(figsize=(16, 8))
           plt.plot(a['Value'], label="Train")
           plt.plot(b['Value'], label="Test")
           plt.plot(c[f'{name}'], label=f'{name}')
           plt.legend(loc="best")
           plt.show()

       results = sm.tsa.ARIMA(data['Value'], order=(2, 1, 2)).fit()
       # predict_sunspots = results.forecast(15)
       predict_sunspots = results.predict(start=str('1991-12-31'), end=str('2020-12-31'), dynamic=False)

       test_arima = test.copy()
       test_arima.insert(2, 'arima', predict_sunspots)

       # fit2 = sm.tsa.ARIMA(data['Value'], order= (2,1,2)).fit()
       # test_arima['arima'] = fit2.predict(start = str('2006-12-31'), end = str('2025-12-31'), dynamic = False)

       test_arima['pre_data'] = test_arima['arima'].shift(axis=0, periods=-1)
       print(test_arima)
       showplot(train, test, test_arima, "pre_data")

ARIMA_predict().run()