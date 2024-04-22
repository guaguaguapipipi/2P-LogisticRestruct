import matplotlib.pyplot as plt
import numpy as np


x = np.array([i for i in range(1, 301)])
print(x)
ay = np.random.rand(300)
print(ay)
by = np.random.rand(300)
cy = np.random.rand(300)
dy = np.random.rand(300)

def with2xLines(x, ay, by, cy, dy):
    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(111)
    ax.plot(x, ay, '#067fce', linewidth=2.0)
    ax.plot(x, by, '#87418f', linewidth=2.0)
    ax.plot(x, cy, '#edc54f', linewidth=2.0)
    ax.plot(x, dy, '#dc552b', linewidth=2.0)

    ax.axhline(max(ay), color='#067fce', linestyle='--')
    ax.axhline(min(ay), color='#067fce', linestyle='--')

    ax.axhline(max(by), color='#87418f', linestyle='--')
    ax.axhline(min(by), color='#87418f', linestyle='--')

    ax.axhline(max(cy), color='#edc54f', linestyle='--')
    ax.axhline(min(cy), color='#edc54f', linestyle='--')

    ax.axhline(max(dy), color='#dc552b', linestyle='--')
    ax.axhline(min(dy), color='#dc552b', linestyle='--')

    plt.show()

with2xLines(x, ay, by, cy, dy)