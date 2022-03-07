import yaml
import torch
from model.nn.layers import Conv2d, LSTM, Linear

class MOSNet(torch.nn.Module):
    def __init__(self, conf):
        super(MOSNet, self).__init__()
        in_channels = 1
        conv_conf = conf['conv_encoder']
        self.conv_encoder = torch.nn.ModuleList()
        for out_channels, activation, kernel, stride in zip(conv_conf['channel'], conv_conf['activations'], conv_conf['kernel'], conv_conf['stride']):
            self.conv_encoder.append(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=(1, stride), padding=(kernel-1) // 2))
            if activation == 'relu':
                self.conv_encoder.append(torch.nn.ReLU())
            in_channels = out_channels

        in_channels *= 4
        self.lstm_encoder = torch.nn.ModuleList()
        lstm_conf = conf['lstm_encoder']
        for hidden, bidirectional in zip(lstm_conf['hidden_size'], lstm_conf['bidirectional']):
            self.lstm_encoder.append(LSTM(input_size=in_channels, hidden_size=hidden, bidirectional=bidirectional, batch_first=True))
            in_channels = hidden

        in_dim = in_channels * 2
        self.linear = torch.nn.ModuleList()
        linear_conf = conf['decoder']
        for hidden, activation, dropout in zip(linear_conf['hidden_size'], linear_conf['activations'], linear_conf['dropout']):
            self.linear.append(Linear(in_features=in_dim, out_features=hidden))
            if activation == 'relu':
                self.linear.append(torch.nn.ReLU())
            if dropout > 0:
                self.linear.append(torch.nn.Dropout(dropout))
            in_dim = hidden
        
        self.l2_criterion = torch.nn.MSELoss()
        self.frame = conf['criterion']['frame']

        self.update_modules = torch.nn.ModuleList([self.conv_encoder, self.lstm_encoder, self.linear])
    
    # override
    def parameters(self):
        return self.update_modules.parameters()

    def forward(self, batchxs, batchys):
        """ 
        input:
            x[0]: spectrogram (B, T, F)
            y: label
        output:
            loss
        """
        x = batchxs[0].data
        y = batchys[0].data

        x = x.unsqueeze(1)
        for layer in self.conv_encoder:
            x = layer(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], 512)

        for layer in self.lstm_encoder:
            x = layer(x)[0]

        for layer in self.linear:
            x = layer(x)

        loss = 0
        if self.frame:
            _y = y.reshape(-1, 1, 1)
            _y = _y.expand(x.shape).clone()
            frame_loss = torch.mean((x-_y) ** 2)
        x = x.mean(dim=1)
        _y = y.reshape(-1, 1)
        _y = _y.expand(x.shape).clone()
        utt_loss = torch.mean((x-_y)**2)
        if self.frame:
            loss = {
                'overall_loss': frame_loss + utt_loss,
                'frame_loss': frame_loss,
                'utt_loss': utt_loss
            }
        else:
            loss = {
                'overall_loss': utt_loss,
                'utt_loss': utt_loss
            }
        return [x], loss
    def inference(self, batchxs):
        x = batchxs[0].data
        
        x = x.unsqueeze(1)
        for layer in self.conv_encoder:
            x = layer(x)
        x = x.permute(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], 512)

        for layer in self.lstm_encoder:
            x = layer(x)

        x = x.mean(dim=1)
        return [x]
        