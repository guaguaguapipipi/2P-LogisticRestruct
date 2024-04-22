import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0., np.e, 0.01)
print(x)
y11 = np.exp(-x)
y12 = np.exp2(-x)
y21 = np.log(x)
y22 = np.log(2*x)


def doubleYaxis(x1, y11, y12, y21, y22, x2, y31, y32, y41, y42):
    fig = plt.figure(figsize=(6, 2))

    ax11 = fig.add_subplot(121)
    ax11.plot(x1, y11, '#ff7a05', linewidth=2.0)
    ax11.plot(x1, y12, '#ff7a05', linewidth=2.0)
    ax11.set_ylabel('Y values for exp(-x)')
    ax11.set_title("Double Y axis")
    ax12 = ax11.twinx()  # this is the important function
    ax12.plot(x1, y21, '#1f77b4', linewidth=2.0)
    ax12.plot(x1, y22, '#1f77b4', linewidth=2.0)
    ax12.set_xlim([0, np.e])
    ax12.set_ylabel('Y values for ln(x)')
    ax12.set_xlabel('Same X for both exp(-x) and ln(x)')

    ax21 = fig.add_subplot(122)
    ax21.plot(x2, y31, '#ff7a05', linewidth=2.0)
    ax21.plot(x2, y32, '#ff7a05', linewidth=2.0)
    ax21.set_ylabel('Y values for exp(-x)')
    ax21.set_title("Double Y axis")
    ax22 = ax21.twinx()  # this is the important function
    ax22.plot(x2, y41, '#1f77b4', linewidth=2.0)
    ax22.plot(x2, y42, '#1f77b4', linewidth=2.0)
    ax22.set_xlim([0, np.e])
    ax22.set_ylabel('Y values for ln(x)')
    ax22.set_xlabel('Same X for both exp(-x) and ln(x)')
    plt.subplots_adjust(left=None, bottom=None, right=None,
                                      top=None, wspace=1, hspace=None)
    plt.show()




doubleYaxis(x, y11, y12, y21, y22, x, y11, y12, y21, y22)