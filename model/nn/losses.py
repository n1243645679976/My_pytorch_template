from numpy import broadcast
import torch
import torch.nn.functional as F

class batchedMSELoss(torch.nn.MSELoss):
    # refer to: https://pytorch.org/docs/stable/notes/broadcasting.html
    # the broadcast is from back, 
    # this method is writen to broadcast from front
    def forward(self, x, y):
        for i in range(y.dim()):
            assert x.shape[i] == y.shape[i] or y.shape[i] == 1
        for i in range(y.dim(), x.dim()):
            y = y.unsqueeze(-1)
        _, y = torch.broadcast_tensors(x, y)
        return super().forward(x, y)

class batchedCELoss(torch.nn.CrossEntropyLoss):
    # refer to: https://pytorch.org/docs/stable/notes/broadcasting.html
    # the broadcast is from back, 
    # this method is writen to broadcast from front
    def forward(self, x, y):
        y = y.long()
        assert y.shape[0] == x.shape[0]
        for i in range(1, y.dim()):
            assert x.shape[i+1] == y.shape[i] or y.shape[i] == 1

        broadcast_target = [x.shape[0]]
        for shape in x.shape[2:]:
            broadcast_target.append(shape)

        for i in range(y.dim(), x.dim()-1):
            y = y.unsqueeze(-1)

        y = torch.broadcast_to(y, broadcast_target)
        return super().forward(x, y)

def get(*args):
  return torch.arange(torch.prod(torch.tensor(args).float()).long()).reshape(*args).float()

if __name__ == '__main__':
    a = get(2, 10, 3, 4)
    b = get(2, 10).long() # this will broadcast to shape (2, 10, 3, 4)
    q = batchedMSELoss()
    print(q(a, b))

    a = get(2, 10, 3, 4)
    b = get(2, 3).long() # this will broadcast to shape (2, 3, 4)
    q = batchedCELoss()
    print(q(a, b))

    a = get(2, 10, 3, 4)
    b = get(2, 1).long() # this will broadcast to shape (2, 3, 4)
    q = batchedCELoss()
    print(q(a, b))