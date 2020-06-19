import torch
import torch.nn.functional as F
import numpy as np

from data.dataset import MNIST
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

from torch.optim import Adam

class VAE:
    def __init__(self, in_dim, encoder_width, decoder_width, latent_dim, device=torch.device('cpu')):
        # device
        self.name = 'VAE'
        self.device = device
        
        # initialize encoder/decoder weights and biases
        self.weights, self.biases = self.init_vae_params(in_dim, encoder_width, decoder_width, latent_dim)
        
        # config dataset
        mnist = MNIST()
        self.train_data = mnist.get(train=True) 
        self.test_data = mnist.get(train=False)

    def train(self, batch_size, max_epoch, lr, weight_decay):
        optimizer = self._get_optimizer(lr, weight_decay)
        hist_loss = []
        
        train_dataloader = DataLoader(
            self.train_data,
            batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
        # print initial loss
        data, _ = next(iter(train_dataloader)) # ingore label
        Xground = data.view((batch_size, -1)).to(self.device)
        loss = self._vae_loss(Xground)
        
        print('initial var_loss = %.5f' % (loss.item()))

        for epoch in range(max_epoch):
            for ii, (data, label) in tqdm(enumerate(train_dataloader)):
                Xground = data.view((batch_size, -1)).to(self.device)
                optimizer.zero_grad()
                loss = self._vae_loss(Xground)
                # backward propagate
                loss.backward()
                optimizer.step()
                hist_loss.append(loss.item())
                
            print('  - epoch #%d, var_loss=%.5f' % (epoch, hist_loss[-1]))
            
        return np.array(hist_loss)
    
    def test1(self, batch_size):
        """ data reconstruction test """
        test_dataloader = DataLoader(
            self.train_data,
            batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
        data, _ = next(iter(test_dataloader))
        Xground = data.view((batch_size, -1)).to(self.device)
        
        z_mean, z_logstd = self._encoding(Xground)
        epsi = torch.randn(z_logstd.size()).to(self.device)
        z_star = z_mean + torch.exp(0.5*z_logstd) * epsi # reparameterize trick
        
        Xstar = self._decoding(z_star)
        Xstar = torch.sigmoid(Xstar)
        
        Xstar = Xstar.view(data.size())
        
#         print(Xstar.size())
        
        return data, Xstar
        
    def test2(self, batch_size):
        """ distribution transformation test(generate artificial dataset from random noises)"""
        pass
        
            
            
        
    def _vae_loss(self, Xground):
        """ compute VAE loss = kl_loss + likelihood_loss """
        
        # KL loss
        z_mean, z_logstd = self._encoding(Xground)
        kl_loss = 0.5*torch.sum(1 + z_logstd - z_mean**2 - torch.exp(z_logstd), dim=1)
        # likelihood loss
        epsi = torch.randn(z_logstd.size()).to(self.device)
        z_star = z_mean + torch.exp(0.5*z_logstd) * epsi # reparameterize trick
        Xstar = self._decoding(z_star)
        
        llh_loss = Xground*torch.log(1e-12 + Xstar) + (1-Xground)*torch.log(1e-12+1-Xstar)
        llh_loss = torch.sum(llh_loss, dim=1)
        
        var_loss = -torch.mean(kl_loss+llh_loss)
        
        return var_loss
    
    def _get_optimizer(self, lr, weight_decay):
        opt_params = []
        # adding weights to optimization paramters list
        for k,v in self.weights.items():
            opt_params.append({'params':v, 'lr':lr})
        # adding biases to optimization parameters list
        for k,v in self.biases.items():
            opt_params.append({'params':v, 'lr':lr})
            
        return Adam(opt_params, lr=lr, weight_decay=weight_decay)
        
        
    def _encoding(self, X):
        # Kingma Supplemtary C.2
        output = torch.matmul(X, self.weights['encoder_hidden_w']) + self.biases['encoder_hidden_b']
        output = torch.tanh(output) 
        mean_output = torch.matmul(output, self.weights['latent_mean_w']) + self.biases['latent_mean_b']
        logstd_output = torch.matmul(output, self.weights['latent_std_w']) + self.biases['latent_std_b']
        
        return mean_output, logstd_output
        
    def _decoding(self, Z):
        output = torch.matmul(Z, self.weights['decoder_hidden_w']) + self.biases['decoder_hidden_b']
        output = torch.tanh(output)
        Xstar = torch.matmul(output, self.weights['decoder_out_w']) + self.biases['decoder_out_b']
        
        Xstar = torch.sigmoid(Xstar)
        
        return Xstar
                    

    def init_vae_params(self, in_dim, encoder_width, decoder_width, latent_dim):
        
        weights = {
            'encoder_hidden_w': self.xavier_init(in_dim, encoder_width),
            'latent_mean_w': self.xavier_init(encoder_width, latent_dim),
            'latent_std_w' : self.xavier_init(encoder_width, latent_dim),
            'decoder_hidden_w': self.xavier_init(latent_dim, decoder_width),
            'decoder_out_w': self.xavier_init(decoder_width, in_dim),
        }
        
        biases = {
            'encoder_hidden_b': self.xavier_init(1, encoder_width),
            'latent_mean_b': self.xavier_init(1, latent_dim),
            'latent_std_b' : self.xavier_init(1, latent_dim),
            'decoder_hidden_b': self.xavier_init(1, decoder_width),
            'decoder_out_b': self.xavier_init(1, in_dim),
        }
            
        return weights, biases
    
    def xavier_init(self, in_d, out_d):
        xavier_stddev = np.sqrt(2.0/(in_d + out_d))
        W = torch.normal(size=(in_d, out_d), mean=0.0, std=xavier_stddev, requires_grad=True, device=self.device)
        return W
    
    