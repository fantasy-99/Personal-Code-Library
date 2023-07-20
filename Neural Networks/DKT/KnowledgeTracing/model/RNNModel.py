import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

#RNN
# class DKT(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super(DKT, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
#         self.output_dim = output_dim
#         # 实际样例的维度、 隐藏层神经元数、 层数
#         self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
#         self.fc = nn.Linear(self.hidden_dim, self.output_dim)
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         # 初始时只有隐藏层的参数，而没有输入，所以搞一个全0的输入
#         h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.m))
#         out, hn = self.rnn(x, h0)
#         res = self.sig(self.fc(out))
#         return res

# LSTM
class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        # 实际样例的维度、 隐藏层神经元数、 层数
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # 初始时只有隐藏层的参数，而没有输入，所以搞一个全0的输入
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        out, (hn, cn) = self.lstm(x, (h0, c0))
        res = self.sig(self.fc(out))
        return res