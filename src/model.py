# encoding: utf-8

import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import *



class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, aggregation_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps

        self.weights = nn.Parameter(torch.Tensor(self.input_caps, input_dim, output_dim))
        
        self.aggregation_module = aggregation_module
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)

        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_dim)
  
        v = self.aggregation_module(u_predict)
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


# Multi_Head_Graph_Pooling        
class Multi_Head_Graph_Pooling(nn.Module):
    def __init__(self, num_caps_types, map_size, n_classes, output_dim, add_loop=True, improved=False, bias=True):
        super(Multi_Head_Graph_Pooling, self).__init__()
        self.n_classes = n_classes
        self.num_caps_types = num_caps_types
        self.map_size = map_size
        self.output_dim = output_dim

        coord = np.zeros((map_size, map_size, 2))
        for i in range(map_size):
            for j in range(map_size):
                coord[i][j][0] = i+1
                coord[i][j][1] = j+1

        adj = torch.from_numpy(compute_adjacency_matrix_images(coord)).float()

        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not improved else 2

        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj_buffer = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        self.register_buffer('adj', adj_buffer)

        self.weight = nn.Parameter(torch.Tensor(output_dim, n_classes))
        self.bias = nn.Parameter(torch.Tensor(n_classes))

        uniform(self.weight)
        zeros(self.bias)
        

    def forward(self, u_predict):
        
        x = u_predict.view(len(u_predict)*self.num_caps_types, self.map_size*self.map_size, -1)

        s = torch.matmul(x, self.weight)
        s = torch.matmul(self.adj, s)
        s = s + self.bias

        s = torch.softmax(s, dim=1)
        x = torch.matmul(s.transpose(1, 2), x)

        u_predict = x.view(len(u_predict), -1, self.n_classes, self.output_dim)
        
        v = u_predict.sum(dim=1)/u_predict.size()[2]
        v = squash(v)
        return v
    

class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10, out_dim=784):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, out_dim)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda: mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class GraCapsNet(nn.Module):
    def __init__(self, incap_dim=8, in_c=3, out_c=512, map_size=14, out_dim=784, n_classes=10, reconstructed=False):
        super(GraCapsNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_c, 256, kernel_size=3, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, int(out_c/incap_dim), incap_dim, kernel_size=3, stride=2)
        
        self.num_primaryCaps = int(out_c/incap_dim) * map_size * map_size
                
        aggregation_module = Multi_Head_Graph_Pooling(int(out_c/incap_dim), map_size, n_classes, 16)
        
        self.digitCaps = CapsLayer(self.num_primaryCaps, incap_dim, n_classes, 16, aggregation_module)

        self.reconstructed = reconstructed
        self.caps = None
        
        if reconstructed == True: self.reconstruction_net = ReconstructionNet(16, 10, out_dim=out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primaryCaps(x)
        x = self.digitCaps(x)

        self.caps = x
        probs = x.pow(2).sum(dim=2).sqrt()
        return probs.log()

    def reconstruct(self, target):
        return self.reconstruction_net(self.caps, target)


def load_model(args):
    model = GraCapsNet(in_c=3, map_size=14, out_dim=args.out_dim, reconstructed=args.reconstructed)
      
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))
    
    return model



    






        
