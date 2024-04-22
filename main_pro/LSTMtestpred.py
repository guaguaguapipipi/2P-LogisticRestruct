import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torchkeras
from plotly import graph_objects as go

def create_dataset(data:list, time_step:int):
    arr_x, arr_y = [], []
    for i in range(len(data) - time_step - 1):
        x = data[i: i + time_step]
        y = data[i + time_step]
        arr_x.append(x)
        arr_y.append(y)
    return np.array(arr_x), np.array(arr_y)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=3, batch_first=True)
        self.fc = nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        # x is input, size (batch_size, seq_len, input_size)
        x, _ = self.lstm(x)
        # x is output, size (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]
        x = self.fc(x)
        x = x.view(-1, 1, 1)
        return x

def lstmPredict(df, time_step=5):
    def train_step(model, features, labels):
        # 正向传播求损失
        predictions = model.forward(features)
        loss = loss_function(predictions, labels)
        # 反向传播求梯度
        loss.backward()
        # 参数更新
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    # 训练模型
    def train_model(model, epochs):
        for epoch in range(1, epochs+1):
            list_loss = []
            for features, labels in dl_train:
                lossi = train_step(model,features, labels)
                list_loss.append(lossi)
            loss = np.mean(list_loss)
            if epoch % 10 == 0:
                print('epoch={} | loss={} '.format(epoch,loss))

    X, Y = create_dataset(df[predict_field].values, time_step)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X.reshape(-1, time_step, 1), dtype=torch.float).to(device)
    Y = torch.tensor(Y.reshape(-1, 1, 1), dtype=torch.float).to(device)
    print('Total datasets: ', X.shape, '-->', Y.shape)
    split_ratio = 0.8
    len_train = int(X.shape[0] * split_ratio)
    X_train, Y_train = X[:len_train, :, :], Y[:len_train, :, :]
    print('Train datasets: ', X_train.shape, '-->', Y_train.shape)
    batch_size = 10
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0)
    ds_train = TensorDataset(X_train, Y_train)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0)

    # 查看第一个batch
    x, y = next(iter(dl_train))
    print(x.shape)
    print(y.shape)
    torch.Size([10, 8, 1])
    torch.Size([10, 1, 1])

    # 自定义训练方式
    model = Net().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # 测试一个batch
    features, labels = next(iter(dl_train))
    loss = train_step(model, features, labels)
    print(loss)

    train_model(model, 100)

    # 预测验证预览
    y_true = Y.cpu().numpy().squeeze()
    y_pred = model.forward(X).detach().cpu().numpy().squeeze()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, name='y_true'))
    fig.add_trace(go.Scatter(y=y_pred, name='y_pred'))
    fig.show()

df = pd.read_csv("转置的数据.csv")


scaler = MinMaxScaler()
predict_field = 'Australia'
df[predict_field] = scaler.fit_transform(df[predict_field].values.reshape(-1, 1))
df = df.iloc[:100]
lstmPredict(df)


