import torch
inf = 1e10
class coattention(torch.nn.Module):
    def __init__(self, xlen1, xlen2):
        """
        xlen1: B*1
        xlen2: B*1
        """
        super(coattention, self).__init__()
        xlen1 = xlen1.long()
        xlen2 = xlen2.long()
        l1max = torch.max(xlen1)
        l2max = torch.max(xlen2)
        self.mask = torch.zeros(xlen1.shape[0], l1max, l2max)
        for i, (l1, l2) in enumerate(zip(xlen1, xlen2)):
            self.mask[i,:l1,:l2] = 1
    def forward(self, x1, x2):
        self.mask = self.mask.to(x1.device)
        x2 = x2.transpose(1,2)
        CoAtt = torch.bmm(x1, x2)
        CoAtt = CoAtt * self.mask - (1-self.mask) * inf
        return CoAtt

if __name__ == '__main__':
    xlen1 = [3,4,6]
    xlen2 = [4,5,3]
    xlen1 = torch.tensor(xlen1).reshape(-1, 1)
    xlen2 = torch.tensor(xlen2).reshape(-1, 1)
    x1 = []
    x2 = []
    for i in [3,4,6]:
        x1.append(torch.randn(i, 80))
    for i in [4,5,3]:
        x2.append(torch.randn(i, 80))
    
    print([_x1.shape for _x1 in x1])
    x1 = torch.nn.utils.rnn.pad_sequence(x1, batch_first=True)
    x2 = torch.nn.utils.rnn.pad_sequence(x2, batch_first=True)
    cat = coattention(xlen1, xlen2)
    cat_map = cat(x1, x2)
    print(x1.shape, cat_map.shape)

