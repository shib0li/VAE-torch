# coding:utf8

import torch
import torch.nn.functional as F
import numpy as np

from data.dataset import MNIST
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

from torch.optim import Adam

class VanillaNet:
    def __init__(self, layers, activation='relu', device=torch.device('cpu')):
        # device
        self.device = device
        
        # config nn structure
        self.layers = layers
        self.num_layers = len(layers)
        self.activation = activation
        self.name = 'Vanilla-Net'
        
        # config learnable variables
        self.activation = {'relu':F.relu, 'sigmoid': torch.sigmoid, 'tanh': F.tanh}[activation]            
        self.weights, self.biases = self.init_nn_weights()
        
        # config dataset
        mnist = MNIST()
        self.train_data = mnist.get(train=True) 
        self.test_data = mnist.get(train=False)
        
        self.criterion = torch.nn.CrossEntropyLoss()

        
    def train(self, batch_size, max_epoch, lr, weight_decay):
        optimizer = self.get_optimizer(lr, weight_decay)
        loss_meter = meter.AverageValueMeter()
        
        train_dataloader = DataLoader(
            self.train_data, 
            batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=0)
        
        for epoch in range(max_epoch):
            loss_meter.reset()
            
            for ii, (data, label) in tqdm(enumerate(train_dataloader)):
                input = data.view((batch_size, -1)).to(self.device)
                target = label.to(self.device)
                
                pred = self.forward(input)
                
                optimizer.zero_grad()
                
                loss = self.criterion(pred, target)
                loss.backward()
                
                optimizer.step()
                
                loss_meter.add(loss.item())

            print('  - epoch #%d, loss=%.5f, test_acc=%.5f' % (epoch, loss_meter.value()[0], self.test()))
                    
    def test(self):
        batch_size = len(self.test_data)
        test_iter = iter(DataLoader(
            self.test_data, 
            batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=0))
        
        data, label = next(test_iter)
        input = data.view((batch_size, -1)).to(self.device)
        target = label.to(self.device)
        
        pred = self.forward(input)
        pred_softmax = torch.nn.Softmax(dim=1)(pred)
        
        pred_cat = torch.argmax(pred_softmax, dim=1)
        
        diff = (pred_cat - target).cpu().numpy().astype(int)
        acc = np.sum((diff==0).astype(int)) / batch_size
        
        return acc

    def init_nn_weights(self):
        weights = []
        biases = []
        
        for l in range(self.num_layers - 1):
            W = self.xavier_init(self.layers[l], self.layers[l+1])
            b = torch.zeros(size=(1, self.layers[l+1]), requires_grad=True, device=self.device)
            weights.append(W)
            biases.append(b)  
            
        return weights, biases
    
    def xavier_init(self, in_d, out_d):
        xavier_stddev = np.sqrt(2.0/(in_d + out_d))
        W = torch.normal(size=(in_d, out_d), mean=0.0, std=xavier_stddev, requires_grad=True, device=self.device)
        return W
    
    def forward(self, X):
        H = X
        for l in range(self.num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = self.activation(torch.mm(H, W) + b)
            
        W = self.weights[-1]
        b = self.biases[-1]
        
        Y = torch.mm(H, W) + b
        return Y
     
    def get_optimizer(self, lr, weight_decay):
        
        parameters = []
        for l in range(self.num_layers - 1):
            parameters.append({'params': self.weights[l], 'lr': lr})
            parameters.append({'params': self.biases[l], 'lr':lr})
        
        return Adam(parameters, lr=lr, weight_decay=weight_decay)

# import torch
# from torch import nn
# from models.generic import GenericNet
# from torch.optim import Adam

# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import numpy as np

# from torch.optim import Adam
        

# class VanillaNet:
#     def __init__(self, layers, activation='relu', device=torch.device('cpu')):
        
#         self.device = device
        
#         self.layers = layers
#         self.num_layers = len(layers)
#         self.activation = activation
#         self.name = 'Vanilla-Net'
        
#         if activation == 'relu':
#             self.activation = F.relu
#         elif activation == 'sigmoid':
#             self.activation = torch.sigmoid
#         elif activation == 'tanh':
#             self.activation = F.tanh
#         else:
#             self.activation = F.relu
            
#         self.weights = []
#         self.biases = []
        
#         for l in range(self.num_layers - 1):
#             W = self.xavier_init(self.layers[l], self.layers[l+1])
#             b = torch.zeros(size=(1, layers[l+1]), requires_grad=True, device=self.device)
#             self.weights.append(W)
#             self.biases.append(b)
    
#     def forward(self, X):
#         H = X
#         for l in range(self.num_layers - 2):
#             W = self.weights[l]
#             b = self.biases[l]
#             H = self.activation(torch.mm(H, W) + b)
            
#         W = self.weights[-1]
#         b = self.biases[-1]
        
#         Y = torch.mm(H, W) + b
#         return Y
     
#     def xavier_init(self, in_d, out_d):
#         xavier_stddev = np.sqrt(2.0/(in_d + out_d))
#         W = torch.normal(size=(in_d, out_d), mean=0.0, std=xavier_stddev, requires_grad=True, device=self.device)
#         return W
    
    
#     def get_optimizer(self, lr, weight_decay):
        
#         parameters = []
#         for l in range(self.num_layers - 1):
#             parameters.append({'params': self.weights[l], 'lr': lr})
#             parameters.append({'params': self.biases[l], 'lr':lr})
        
#         return Adam(parameters, lr=lr, weight_decay=weight_decay)
    
# in_d = 5
# out_d = 10
# hidden_width = 128
# hidden_depth = 3


# # layers = [in_d] + [hidden_width]*hidden_depth + [out_d]
# layers = [in_d] + [128,129,130] + [out_d]

# Ntrain = 39
# X = torch.randn(size=(Ntrain, in_d))


# model = VanillaNet(layers, activation='sigmoid')

# model.forward(X)
    

# class VanillaNet(GenericNet):
#     def __init__(self, layers, activation):
#         super(VanillaNet, self).__init__()
#         self.layers = layers
        
#         self.model_name = 'VanilaNet'
        
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'sigmoid':
#             self.activation = nn.Sigmoid()
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         else:
#             self.activation = nn.ReLU()
            
#         self.net = self.build()
        
#     def build(self):

#         # build nerual net
#         net = nn.Sequential()
#         for i in range(len(self.layers)-2):
#             in_d = self.layers[i]
#             out_d = self.layers[i+1]
            
#             linear_name = 'linear_' + str(i)
#             activation_name = 'activation_' + str(i)
            
#             net.add_module(linear_name, nn.Linear(in_d, out_d))
#             net.add_module(activation_name, self.activation)
        
#         # last linear projection layer
#         net.add_module('last_linear', nn.Linear(self.layers[-2], self.layers[-1]))
        
#         return net
            
#     def forward(self, X):
#         return self.net(X)
    
#     def get_optimizer(self, lr, weight_decay):
#         return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)