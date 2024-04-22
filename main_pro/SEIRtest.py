import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as spi
import pylab as pl
import math

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

N = 5000000  # 人口总数
E = []  # 潜伏携带者
E.append(0)
I = []  # 传染者
I.append(1)
S = []  # 易感者
S.append(N - I[0])
R = []  # 康复者
R.append(0)

D = []  # 死亡者
D.append(0)

r = 10  # 传染者接触人数
b = 0.03  # 传染者传染概率
a = 0.1  # 潜伏者患病概率
r2 = 30  # 潜伏者接触人数
b2 = 0.03  # 潜伏者传染概率
y = 0.2  # 康复概率
k = 0.01  # 死亡率
T = [i for i in range(0, 150)]  # 时间


def calc(T):
    global r
    global r2
    for i in range(0, len(T) - 1):
        if i >= 27:
            r = 2
            r2 = 2

        S.append(S[i] - r * b * S[i] * I[i] / N - r2 * b2 * S[i] * E[i] / N)
        E.append(E[i] + r * b * S[i] * I[i] / N - a * E[i] + r2 * b2 * S[i] * E[i] / N)
        I.append(I[i] + a * E[i] - (k + y) * I[i])
        R.append(R[i] + y * I[i])
        D.append(D[i] + k * I[i])


def plot(T, S, E, I, R, D):
    plt.figure()
    plt.title("SEIR-nCoV 传播时间曲线")
    plt.plot(T, S, color='r', label='易感者')
    plt.plot(T, E, color='k', label='潜伏者')
    plt.plot(T, I, color='b', label='传染者')
    plt.plot(T, R, color='g', label='移除者')
    plt.plot(T, D, color='black', label='死亡者', linestyle='dashdot')
    plt.grid(False)
    plt.legend()
    plt.xlabel("时间(天)")
    plt.ylabel("人数")
    pl.savefig('SEIR-nCoV 传播时间曲线.png', dpi=900)
    plt.show()


if __name__ == '__main__':
    calc(T)
    plot(T, S, E, I, R, D)


