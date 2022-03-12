import torch
import os
import sys
import inspect
sys.path.append(os.getcwd())
from model.nn.layers import Conv1d, LSTM, Linear
from model.nn.coattention import coattention
from model.nn.lib_sincnet import SincNet
import torch.nn.functional as F

class batched_data():
    def __init__(self, data, data_len):
        self.data = data
        self.len = data_len
    def __repr__(self):
        return f'data shape: {self.data.shape}, length of data: {self.len}'

class ResidualBlockW(torch.nn.Module):
    def __init__(self, dilation):
        super(ResidualBlockW, self).__init__()
        input_channel = 64
        skipped_channel = 64
        gate_channel = 128
        res_channel = 64
        self.input_conv = Conv1d(input_channel, gate_channel, kernel_size=3, dilation=dilation, padding=0)
        self.skipped_conv = Conv1d(gate_channel // 2, skipped_channel, kernel_size=1)
        self.res_conv = Conv1d(gate_channel // 2, res_channel, kernel_size=1)
        self.gc = gate_channel
        self.padding = dilation * 2
    def forward(self, x):
        res = x
        x = F.pad(x, (0, self.padding))
        gate_x = self.input_conv(x)
        xt, xs = torch.split(gate_x, self.gc // 2, dim=1)
        out = torch.tanh(xt) * torch.sigmoid(xs)
        s = self.skipped_conv(out)
        x = self.res_conv(out)
        x += res
        x *= 0.707
        return x, s

class WaveResNet(torch.nn.Module):
    def __init__(self, input_channel=64, skipped_channel=64):
        super(WaveResNet, self).__init__()
        layers = 7
        stacks = 1
        lps = layers // stacks
        self.first_conv = Conv1d(input_channel, skipped_channel, kernel_size=1)
        self.conv = torch.nn.ModuleList([])
        for layer in range(layers):
            dilation = 2 ** (layer % lps)
            self.conv.append(ResidualBlockW(dilation=dilation))
        self.last_conv = torch.nn.Sequential(
                torch.nn.ReLU(),
                Conv1d(skipped_channel, skipped_channel, kernel_size=1),
                torch.nn.ReLU(),
                Conv1d(skipped_channel, skipped_channel, kernel_size=1)
                )
    def forward(self, x):
        x = self.first_conv(x)
        s = 0
        for layer in self.conv:
            x, _s = layer(x)
            s += _s
        y = self.last_conv(s)
        return y

class regression_model(torch.nn.Module):
    def __init__(self, conf):
        super(regression_model, self).__init__()
        self.sincnet = SincNet(N_cnn_lay=1)
        self.wavenet = torch.nn.ModuleList(
                        [WaveResNet() for i in range(4)])
        self.downsample = torch.nn.ModuleList(
                        [torch.nn.MaxPool1d(kernel_size=4),
                         torch.nn.MaxPool1d(kernel_size=5),
                         torch.nn.MaxPool1d(kernel_size=2),
                         torch.nn.MaxPool1d(kernel_size=2)])
        self.rnn = LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True, num_layers=1)
        self.derive_model = torch.nn.Sequential(
                                 Linear(256, 128),
                                 torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                 Linear(128, 1)
                                 )
                                 
    def encode_frame_embedding(self, x, frame_lengths):
        y = self.sincnet(x)
        for i in range(4):
            y = self.wavenet[i](y)
            y = self.downsample[i](y)
        y = torch.nn.utils.rnn.pack_padded_sequence(y.transpose(1,2), frame_lengths.reshape(-1), batch_first=True, enforce_sorted=False)
        y = self.rnn(y)[0]
        y = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)[0]
        return y
    def encode_frame_score(self, x):
        y = self.encode_frame_embedding(x)
        y = [self.derive_model(_y) for _y in y]
        return y
    def forward(self, batchxs):
        x1 = batchxs[0].data   # B*T
        x2 = batchxs[1].data   # B*T
        lenX1 = batchxs[0].len # B*1
        lenX2 = batchxs[1].len # B*1
        outputs = []

        # downsample by sincnet
        frame_numX1 = torch.div(lenX1 - 250, 3, rounding_mode='floor')
        frame_numX2 = torch.div(lenX2 - 250, 3, rounding_mode='floor')
        # downsample by maxpooling
        for d in self.downsample:
            frame_numX1 = torch.div(frame_numX1, d.kernel_size, rounding_mode='floor')
            frame_numX2 = torch.div(frame_numX2, d.kernel_size, rounding_mode='floor')

        y1 = self.encode_frame_embedding(x1, frame_numX1) # BTF
        y2 = self.encode_frame_embedding(x2, frame_numX2) # BTF

        for i, (l1, l2) in enumerate(zip(frame_numX1, frame_numX2)):
            y1[i, :, l1:] = 0
            y2[i, :, l2:] = 0
        
        cat = coattention(frame_numX1, frame_numX2)
        cat_map = cat(y1, y2)

        atty1 = torch.softmax(cat_map, dim=2).bmm(y2)
        atty2 = torch.softmax(cat_map, dim=1).transpose(1, 2).bmm(y1)

        for i, (l1, l2) in enumerate(zip(frame_numX1, frame_numX2)):
            atty1[i, l1:] = 0
            atty2[i, l2:] = 0
        

        diffy1 = torch.abs(atty1.sum(dim=1)-y1.sum(dim=1))
        diffy2 = torch.abs(atty2.sum(dim=1)-y2.sum(dim=1))
        for i, (l1, l2) in enumerate(zip(frame_numX1, frame_numX2)):
            diffy1[i] /= l1
            diffy2[i] /= l2

        y1 = self.derive_model(diffy1)
        y2 = self.derive_model(diffy2)
        y = (y1 + y2) / 2

        outputs.append(y)
        return outputs
    def predict(self, x, label):
        with torch.no_grad():
            return self.forward(x, label)


class classfication_model(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(classfication_model, self).__init__(device)
        self.sincnet = SincNet(N_cnn_lay=1, device=device)
        self.wavenet = torch.nn.ModuleList(
                        [WaveResNet() for i in range(4)])
        self.downsample = torch.nn.ModuleList(
                        [torch.nn.MaxPool1d(kernel_size=4),
                         torch.nn.MaxPool1d(kernel_size=5),
                         torch.nn.MaxPool1d(kernel_size=2),
                         torch.nn.MaxPool1d(kernel_size=2)])
        self.rnn = LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True, num_layers=1)
        self.derive_model = torch.nn.Sequential(
                                Linear(256, 128),
                                torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                Linear(128, 4)
                                )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.to(device)
        self.device=device

    def encode_frame_embedding(self, x, frame_lengths):
        y = self.sincnet(x)
        for i in range(4):
            y = self.wavenet[i](y)
            y = self.downsample[i](y)
        y = torch.nn.utils.rnn.pack_padded_sequence(y.transpose(1,2), frame_lengths.reshape(-1), batch_first=True, enforce_sorted=False)
        y = self.rnn(y)[0]
        y = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)[0]
        return y
    def encode_frame_score(self, x):
        y = self.encode_frame_embedding(x)
        y = [self.derive_model(_y) for _y in y]
        return y

    def forward(self, batchxs):
        x1 = batchxs[0].data   # B*T
        x2 = batchxs[1].data   # B*T
        lenX1 = batchxs[0].len # B*1
        lenX2 = batchxs[1].len # B*1
        outputs = []

        # downsample by sincnet
        frame_numX1 = torch.div(lenX1 - 250, 3, rounding_mode='floor')
        frame_numX2 = torch.div(lenX2 - 250, 3, rounding_mode='floor')
        # downsample by maxpooling
        for d in self.downsample:
            frame_numX1 = torch.div(frame_numX1, d.kernel_size, rounding_mode='floor')
            frame_numX2 = torch.div(frame_numX2, d.kernel_size, rounding_mode='floor')

        y1 = self.encode_frame_embedding(x1, frame_numX1) # BTF
        y2 = self.encode_frame_embedding(x2, frame_numX2) # BTF

        for i, (l1, l2) in enumerate(zip(frame_numX1, frame_numX2)):
            y1[i, :, l1:] = 0
            y2[i, :, l2:] = 0
        
        cat = coattention(frame_numX1, frame_numX2)
        cat_map = cat(y1, y2)

        atty1 = torch.softmax(cat_map, dim=2).bmm(y2)
        atty2 = torch.softmax(cat_map, dim=1).transpose(1, 2).bmm(y1)

        for i, (l1, l2) in enumerate(zip(frame_numX1, frame_numX2)):
            atty1[i, l1:] = 0
            atty2[i, l2:] = 0
        

        diffy1 = torch.abs(atty1.sum(dim=1)-y1.sum(dim=1))
        diffy2 = torch.abs(atty2.sum(dim=1)-y2.sum(dim=1))
        for i, (l1, l2) in enumerate(zip(frame_numX1, frame_numX2)):
            diffy1[i] /= l1
            diffy2[i] /= l2

        y1 = self.derive_model(diffy1)
        y2 = self.derive_model(diffy2)
        y = (y1 + y2) / 2
        y = y.reshape(-1, 4)

        outputs.append(y)
        return outputs




if __name__ == '__main__':
    import numpy as np
    rm = regression_model('cpu').eval()
    len1 = [3000, 2000, 1500]
    tensor_len1 = torch.tensor(len1)
    x1 = torch.randn(3, np.max(len1))
    x1[1] *= 10
    x1[2] *= 100
    for l in range(len(len1)):
        x1[l, len1[l]:] = 0
    
    pert_x1 = []
    pert_len1 = []
    for i in [2, 0, 1]:
        pert_x1.append(x1[i].unsqueeze(0))
        pert_len1.append(len1[i] + 5000 if i == 2 else len1[i])
    pert_len1 = torch.tensor([1500, 8000, 2000])
    pert_x1 = torch.cat(pert_x1, dim=0)
    print(pert_x1.shape)
    pert_x1 = torch.cat([pert_x1, torch.zeros(pert_x1.shape[0], 5000)], dim=1)


    len_target = [3000, 2000, 1500]
    tensor_len_target = torch.tensor(len_target)
    target_x = torch.randn(3, np.max(len_target))
    for l in range(len(len_target)):
        target_x[l, len_target[l]:] = 0
    

    pert_target = []
    pert_len_target = []
    for i in [2, 0, 1]:
        pert_target.append(target_x[i].unsqueeze(0))
    pert_len_target = torch.tensor([1500, 3000, 2000])
    pert_target = torch.cat(pert_target, dim=0)

    print(rm([batched_data(x1, tensor_len1), batched_data(target_x, tensor_len_target)]))
    print(rm([batched_data(pert_x1, pert_len1), batched_data(pert_target, pert_len_target)]))