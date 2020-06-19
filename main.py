from config import opt
import os
import torch
import models
from data.dataset import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
import fire

from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image
import numpy as np


def evaluation(**kwargs):
   
    opt._parse(kwargs)
    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

    # config model
    model = models.VAE(opt.in_d, opt.encoder_hidden_width, opt.decoder_hidden_width, opt.latent_dim, device)
    # train VAE
    model.train(opt.batch_size, opt.max_epoch, opt.lr, opt.weight_decay)

    test_batch = 64
    ground, Xstar = model.test1(test_batch)


    img_ground = make_grid(ground, int(np.sqrt(test_batch)), normalize=True)
    save_image(img_ground, 'ground.png')
    
# Image.open('ground.png')

    img_pred = make_grid(Xstar, int(np.sqrt(test_batch)), normalize=True)

    save_image(img_pred, 'pred.png')
# Image.open('ground.png')
                

if __name__=='__main__':
    fire.Fire()
