import numpy as np
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



def evaluation(**kwargs):
   
    opt._parse(kwargs)
    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

    # config model
    model = models.VAE(opt.in_d, opt.encoder_hidden_width, opt.decoder_hidden_width, opt.latent_dim, device)
    # train VAE
    hist_loss = model.train(opt.batch_size, opt.max_epoch, opt.lr, opt.weight_decay)
    np.savetxt('figures/loss.csv', hist_loss, delimiter=',')

    test_batch = 64
    
    # Experiment 1: recover the original images
    ground, Xstar = model.test1(test_batch)
    img_ground = make_grid(ground, int(np.sqrt(test_batch)), normalize=True)
    save_image(img_ground, 'figures/ground.png')

    img_recover = make_grid(Xstar, int(np.sqrt(test_batch)), normalize=True)
    save_image(img_recover, 'figures/recover.png')
    
    # Experiment 2: generate artificial images from noise
    Xnoise = model.test2(test_batch) # generated images from Gaussian noise
    img_noise = make_grid(Xnoise, int(np.sqrt(test_batch)), normalize=True)
    save_image(img_noise, 'figures/noise.png')
     

if __name__=='__main__':
    fire.Fire()
