import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat=1433):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 512)
        self.drop1 = nn.Dropout(p=0.5)
        self.gc2 = GraphConvolution(512, 128)
        self.drop2 = nn.Dropout(p=0.5)
        self.relu = nn.LeakyReLU(0.1)
        self.linear = nn.Linear(128, 7)
    def forward(self, x, adj):
        x = self.relu(self.gc1(x, adj))
        x = self.drop1(x)
        x = self.relu(self.gc2(x, adj))
        x = self.drop2(x)
        x = self.linear(x)
        return x