# encoding: utf-8

import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from model import load_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='./data/', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    
    parser.add_argument('--model', default='GraCaps', type=str, help='the name of the model')
    parser.add_argument('--reconstructed', action='store_true', default=False)
    parser.add_argument('--reconstruction_alpha', type=float, default=0.005)
    parser.add_argument('--out_dim', default=32*32*3, type=int, help='the dim of reconstructed image')
    parser.add_argument('--loss', default='Marg', type=str, help='loss function')
    
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_schedule', default='constant', type=str, choices=['multistep', 'constant'])
    parser.add_argument('--lr_decay_epochs', default=None, type=str)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--lr', default=0.001, type=float)
    
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--out_dir', default='./model_save/', type=str, help='Output directory')
    
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cpu', type=str, help='the current device')
    return parser.parse_args()


def init_args(args):
    # set learning rate decay epochs
    if args.lr_decay_epochs is not None:
        iterations = args.lr_decay_epochs.split(',')
        args.lr_decay_epochs = list([])
        for it in iterations:
            args.lr_decay_epochs.append(int(it))

    # set up saving name
    args.model_name = '{}_{}_recons_{}_loss_{}_LS_{}_Seed_{}'.format(
        args.dataset, args.model, args.reconstructed, args.loss, args.lr_schedule, args.seed
        )

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create folders
    args.out_dir = os.path.join(args.out_dir, args.model_name)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)        
    return args


# loss function
class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5, Islogit=True):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_
        self.Islogit = Islogit

    def forward(self, lengths, targets, size_average=True):
        if self.Islogit: lengths = torch.exp(lengths)
        
        targets = targets.long()
        if torch.cuda.is_available(): t = torch.cuda.FloatTensor(lengths.size()).fill_(0)
        else: t = torch.zeros(lengths.size()).long()
        
        if len(targets.shape) == 2: t = t.scatter_(1, targets.data.view(-1, 2), 1)
        else: t = t.scatter_(1, targets.data.view(-1, 1), 1)
        
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()


# training
def train_epoch(args, epoch, train_loader, model, criterion, optimizer, scheduler=None):
    train_loss = 0
    train_acc = 0
    train_n = 0
    model.train()
    
    for i, (X, y) in tqdm(enumerate(train_loader)):
        X, y = X.cuda(), y.cuda()
        output = model(X)
        loss = criterion(output, y)

        if args.reconstructed:
            reconstruction_loss = F.mse_loss(model.reconstruct(y), X.view(-1, args.out_dim))
            loss += args.reconstruction_alpha * reconstruction_loss
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

    return train_loss, train_acc, train_n
        

# test
def test(args, test_loader, model, criterion):
    test_loss = 0
    test_acc = 0
    test_n = 0
    model.eval()

    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        output = model(X)
        loss = criterion(output, y)
        
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        test_n += y.size(0)

    print('Test: \t Loss %.4f \t Accu %.4f'%(test_loss/test_n, test_acc/test_n))  
    return test_loss, test_acc, test_n

def load_data(args):
    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()
                                        ])
        test_transform = transforms.Compose([transforms.ToTensor()])
            
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/', train=True, download=True, transform=train_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/', train=False, transform=test_transform),
            batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
    return train_loader, test_loader


def main():
    args = get_args()
    args = init_args(args)
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = load_data(args)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'Marg':
        criterion = MarginLoss()
    
    model = load_model(args).to(args.device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load state
    if args.resume is not None: 
        state = torch.load(args.resume)
        start_epoch = state['start_epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Model Loaded Successfully!')
        
    # evaluation
    if args.evaluate:
        test(args, test_loader, model, criterion)
        return

    # learning rate
    if args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)
    elif args.lr_schedule == 'constant':
        scheduler = None
    else:
        print('Learning scheduler is not supported!')
        
    save_file = os.path.join(args.out_dir, args.model + '.pth')

    # training
    start_train_time = time.time()
    for epoch in range(1, args.epochs+1):
        
        start_epoch_time = time.time()
        train_loss, train_acc, train_n = train_epoch(args, epoch, train_loader, model, criterion, optimizer, scheduler=scheduler)
        epoch_time = time.time()

        if scheduler is not None: scheduler.step()
        
        print('Epoch: %d \t Time: %.2f \t Loss: %.4f \t Accu: %.4f'%(epoch, epoch_time - start_epoch_time, train_loss/train_n, train_acc/train_n))
        test(args, test_loader, model, criterion)

    #save model 
    state = {}
    state['start_epoch']  = epoch
    state['optimizer']    = optimizer.state_dict()
    state['model']  = model.state_dict()
    torch.save(state, save_file)
            
    train_time = time.time()
    print('Total train time: %.4f minutes'%((train_time - start_train_time)/60))

if __name__ == "__main__":
    main()





    
