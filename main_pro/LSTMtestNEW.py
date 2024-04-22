#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tushare as ts
import torch
from torch import nn

DAYS_FOR_TRAIN = 10


class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归

        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x


def create_dataset(data, days_for_train=5) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集

        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。

        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return (np.array(dataset_x), np.array(dataset_y))


if __name__ == '__main__':
    data_close = ts.get_k_data('000001', start='2018-01-01', index=True)[
        'close'].values  # 取上证指数的收盘价的np.ndarray 而不是pd.Series
    data_close = data_close.astype('float32')  # 转换数据类型
    plt.plot(data_close)
    plt.savefig('data.png', format='png', dpi=200)
    plt.close()

    # 将价格标准化到0~1
    max_value = np.max(data_close)
    min_value = np.min(data_close)
    data_close = (data_close - min_value) / (max_value - min_value)

    dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)

    # 划分训练集和测试集，70%作为训练集
    train_size = int(len(dataset_x) * 0.7)

    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]
    # test_x = dataset_x[train_size:]  # 暂时没有用到
    # test_y = dataset_y[train_size:]  # 暂时没有用到

    # 将数据改变形状，RNN 读入的数据维度是 (seq_size, batch_size, feature_size)
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    train_y = train_y.reshape(-1, 1, 1)
    # test_x = test_x.reshape(-1, 1, DAYS_FOR_TRAIN)

    # 转为pytorch的tensor对象
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    # test_x = torch.from_numpy(test_x)

    model = LSTM_Regression(DAYS_FOR_TRAIN, 8, output_size=1, num_layers=2)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for i in range(1000):
        out = model(train_x)
        loss = loss_function(out, train_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))

    # torch.save(model.state_dict(), 'model_params.pkl')  # 可以保存模型的参数供未来使用

    # for test
    model = model.eval()  # 转换成测试模式
    # model.load_state_dict(torch.load('model_params.pkl'))  # 读取参数

    # 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
    dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
    dataset_x = torch.from_numpy(dataset_x)

    pred_test = model(dataset_x)  # 全量训练集的模型输出 (seq_size, batch_size, output_size)
    pred_test = pred_test.view(-1).data.numpy()
    pred_test = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test))  # 填充0 使长度相同
    assert len(pred_test) == len(data_close)

    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(data_close, 'b', label='real')
    plt.plot((train_size, train_size), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    plt.legend(loc='best')
    plt.savefig('result.png', format='png', dpi=200)
    plt.close()


#  在这里做一个额外的判断