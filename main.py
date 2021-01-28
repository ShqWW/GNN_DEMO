import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import GCN

def cal_acc(pre, gt):
    pre = torch.argmax(pre, dim=1)
    return torch.sum(pre==gt)/pre.shape[0]

raw_data = pd.read_csv('./dataset/cora.content', sep = '\t', header = None)
raw_data_cites = pd.read_csv('./dataset/cora.cites', sep = '\t', header = None)
map = dict(zip(list(raw_data[0]), list(raw_data.index)))
features =np.array(raw_data.iloc[:,1:-1])
label = np.argmax(np.array(pd.get_dummies(raw_data[1434]).iloc[:,:]), axis=1)
matrix = np.zeros((raw_data.shape[0], raw_data.shape[0]))
for i ,j in zip(raw_data_cites[0], raw_data_cites[1]):
    x, y = map[i], map[j]
    matrix[x][y] = matrix[y][x] = 1

features = torch.from_numpy(features).float()
adj = torch.from_numpy(matrix).float()
label = torch.from_numpy(label)
net = GCN(nfeat=1433)
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
net.train()

D = torch.sum(adj, dim=1)
index = torch.where(D!=0)[0]
adj[index, :] = adj[index, :] / D[index]
adj = torch.eye(adj.shape[0]) + adj

bound=270*5
for i in range(10):
    optimizer.zero_grad()
    pre = net(features, adj)
    train_pre = pre[0:bound,:]
    train_label = label[0:bound]
    loss = criteria(train_pre, train_label)
    loss.backward()
    optimizer.step()

    net.eval()
    with torch.no_grad():
        pre = net(features, adj)
    net.train()
    train_result = cal_acc(pre[0:bound,:], label[0:bound])
    test_result = cal_acc(pre[bound:,:], label[bound:])
    print(train_result.item(), test_result.item())
    # train_result = cal_acc(pre[0:1000,:], label[0:1000])
    # val_result = cal_acc(pre[1000:1500,:], label[1000:1500])
    # test_result = cal_acc(pre[1500:,:], label[1500:])
    # print(train_result.item(), val_result.item(), test_result.item())
    
        
        
