from config import opt
import os
import torch
import models
from data.dataset import MNIST
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
import fire


def evaluation(**kwargs):
   
    opt._parse(kwargs)
    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

#     # config model
#     layers = [opt.in_d] + [opt.hidden_width]*opt.hidden_depth + [opt.out_d]
#     model = models.VanillaNet(layers, opt.activation, device)

#     model.train(opt.batch_size, opt.max_epoch, opt.lr, opt.weight_decay)
    
#     model.test()
                

if __name__=='__main__':
    fire.Fire()
