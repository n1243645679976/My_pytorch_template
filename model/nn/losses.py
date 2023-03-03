from numpy import broadcast
import torch
import torch.nn.functional as F
from torch import nn

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
        if len(x.shape) == len(y.shape) == 2:
            y = y.squeeze(1)
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


class ContrastiveLoss(nn.Module):
    '''
    Contrastive Loss
    Args:
        margin: non-neg value, the smaller the stricter the loss will be, default: 0.2        
        
    '''
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pred_score, gt_score):
        pred_score = pred_score.squeeze(1)
        gt_score = gt_score.squeeze(1)
        assert len(pred_score.shape) == 1 and len(gt_score.shape) == 1
        # pred_score, gt_score: tensor, [batch_size]  
        gt_diff = gt_score.unsqueeze(1) - gt_score.unsqueeze(0)
        pred_diff = pred_score.unsqueeze(1) - pred_score.unsqueeze(0)
        loss = torch.maximum(torch.zeros(gt_diff.shape).to(gt_diff.device), torch.abs(pred_diff - gt_diff) - self.margin) 
        loss = loss.mean().div(2)
        
        return loss


class ClippedMSELoss(nn.Module):
    """
    clipped MSE loss for listener-dependent model
    """
    def __init__(self, tau, mode='frame'):
        super(ClippedMSELoss, self).__init__()
        self.tau = torch.tensor(tau,dtype=torch.float)

        self.criterion = torch.nn.MSELoss(reduction='none')
        self.mode = mode


    def forward_criterion(self, y_hat, label):

        loss = self.criterion(y_hat, label)
        threshold = torch.abs(y_hat - label) > self.tau
        loss = torch.mean(threshold * loss)
        return loss

    def forward(self, pred_score, gt_score):
        """
        Args:
            pred_mean, pred_score: [batch, time, 1/5]
        """
        # repeat for frame level loss
        time = pred_score.shape[1]
        if self.mode == 'utt':
            pred_score = pred_score.mean(dim=1)
        else:
            gt_score = gt_score.unsqueeze(1)
            gt_score = gt_score.repeat(1, time, 1)
        main_loss = self.forward_criterion(pred_score, gt_score)
        return main_loss # lamb 1.0  

class Focal_MultiLabel_Loss(nn.Module):
    def __init__(self, gamma):
      super(Focal_MultiLabel_Loss, self).__init__()
      self.gamma = gamma
      self.bceloss = batchedCELoss(reduction='none')

    def forward(self, outputs, targets): 
      bce = self.bceloss(outputs, targets)
      bce_exp = torch.exp(-bce)
      focal_loss = (1-bce_exp)**self.gamma * bce
      return focal_loss.mean()

if __name__ == '__main__':
    cme = ClippedMSELoss(tau= 0.25,mode='frame')
    a = get(2, 3, 1)
    b = get(2, 1)
    print(cme(a, b))

    closs = ContrastiveLoss()
    a = get(2, 1)
    b = get(2, 1)
    print(closs(a, b))

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
