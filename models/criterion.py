import torch
import torch.nn as nn


class CELoss(torch.nn.Module):
    def __init__(self, masked=True):
        super().__init__()
        self.celoss = nn.CrossEntropyLoss(reduction='none')
    def forward(self, pred, target):
        loss = self.celoss(pred, target).mean()
        return loss
    
    
    
    
