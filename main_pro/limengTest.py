import torch
from torch import nn
rnn = nn.LSTM(10,20,2)
inget = torch.randn(5,3,10)
h0 = torch.randn(2,3,20)
c0 = torch.randn(2,3,20)
outget, (hn, cn) = rnn(inget, (h0,c0))
print(inget)
print(outget)