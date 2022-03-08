import torch
import os
import sys
import inspect
sys.path.append(os.getcwd())
from model.nn.layers import Conv1d, LSTM, Linear
from model.nn.coattention import coattention
from model.nn.lib_sincnet import SincNet

class ResidualBlockW(torch.nn.Module):
    def __init__(self, dilation):
        super(ResidualBlockW, self).__init__()
        input_channel = 64
        skipped_channel = 64
        gate_channel = 128
        res_channel = 64
        self.input_conv = Conv1d(input_channel, gate_channel, kernel_size=3, dilation=dilation, padding=dilation)
        self.skipped_conv = Conv1d(gate_channel // 2, skipped_channel, kernel_size=1)
        self.res_conv = Conv1d(gate_channel // 2, res_channel, kernel_size=1)
        self.gc = gate_channel

    def forward(self, x):
        res = x
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
        self.lm2emb = torch.nn.Sequential(Linear(768, 256), torch.nn.ReLU(), Linear(256, 256))

    def encode_frame_embedding(self, x):
        y = self.sincnet(x)
        for i in range(4):
            y = self.wavenet[i](y)
            y = self.downsample[i](y)
        y = self.rnn(y.transpose(1,2))[0]
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

        wavlm_feat1 = batchxs[2].data
        wavlm_feat2 = batchxs[3].data
        lenWavLM_feat1 = batchxs[2].len
        lenWavLM_feat2 = batchxs[3].len

        # downsample by sincnet
        frame_numX1 = torch.div(lenX1 - 250, 3, rounding_mode='floor')
        frame_numX2 = torch.div(lenX2 - 250, 3, rounding_mode='floor')
        # downsample by maxpooling
        for d in self.downsample:
            frame_numX1 = torch.div(frame_numX1, d.kernel_size, rounding_mode='floor')
            frame_numX2 = torch.div(frame_numX2, d.kernel_size, rounding_mode='floor')

        y1 = self.encode_frame_embedding(x1) # BTF
        y2 = self.encode_frame_embedding(x2) # BTF
        wavlm_feat1 = self.lm2emb(wavlm_feat1)
        wavlm_feat2 = self.lm2emb(wavlm_feat2)

        newMaxLen1 = torch.max(frame_numX1 + lenWavLM_feat1)
        newMaxLen2 = torch.max(frame_numX2 + lenWavLM_feat2)
        newY1 = torch.zeros(y1.shape[0], newMaxLen1, y1.shape[2]).to(y1.device)
        newY2 = torch.zeros(y2.shape[0], newMaxLen2, y2.shape[2]).to(y1.device)

        for i in range(y1.shape[0]):
            newY1[i, :frame_numX1[i]] = y1[i, :frame_numX1[i]]
            newY1[i, frame_numX1[i]:frame_numX1[i]+lenWavLM_feat1[i]] = wavlm_feat1[i, :lenWavLM_feat1[i]]
            newY2[i, :frame_numX2[i]] = y2[i, :frame_numX2[i]]
            newY2[i, frame_numX2[i]:frame_numX2[i]+lenWavLM_feat2[i]] = wavlm_feat2[i, :lenWavLM_feat2[i]]

        y1 = newY1
        y2 = newY2

        frame_numX1 += lenWavLM_feat1
        frame_numX2 += lenWavLM_feat2

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

    def encode_frame_embedding(self, x):
        y = [self.sincnet(_x) for _x in x]
        for i in range(4):
            y = [self.wavenet[i](_y) for _y in y]
            y = [self.downsample[i](_y) for _y in y]
        y = [self.rnn(_y.transpose(1,2))[0] for _y in y]
        return y
    def encode_frame_score(self, x):
        y = self.encode_frame_embedding(x)
        y = [self.derive_model(_y) for _y in y]
        return y

    def forward(self, batchxs, batchys):
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
            frame_numX1 //= d.kernel_size
            frame_numX2 //= d.kernel_size

        y1 = self.encode_frame_embedding(x1) # BTF
        y2 = self.encode_frame_embedding(x2) # BTF

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

class batched_data():
    def __init__(self, data, data_len):
        self.data = data
        self.len = data_len
    def __repr__(self):
        return f'data shape: {self.data.shape}, length of data: {self.len}'

if __name__ == '__main__':
    import numpy as np
    rm = regression_model('cpu')
    x1len = [17664, 4801, 7972]
    x1 = torch.randn(3, np.max(x1len))
    x2len = [500, 7349, 2950]
    x2 = torch.randn(3, np.max(x2len))
    print(rm([batched_data(x1, torch.tensor(x1len).long().reshape(-1, 1)), batched_data(x2, torch.tensor(x2len).long().reshape(-1, 1))])[0].shape)