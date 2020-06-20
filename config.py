import warnings
import torch

class DefaultConfig(object):

    # default model
    model = 'VAE'
    encoder_hidden_width = 1024
    decoder_hidden_width = 1024
    latent_dim = 256 # size of latent vector
    in_d = 28*28
    
    load_model_path = None
    
    # train config
    batch_size = 1000
    use_gpu = True
    num_workers = 0
    print_freq = 100
    
    result_fname = 'result.csv'
    
    max_epoch = 10
    lr = 1e-3
    lr_decay = 0.5
    weight_decay = 0e-5

    def _parse(self, kwargs):
        for k,v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('WARNING: options does not include attribute %s' % k)
            setattr(self, k, v)
        
        print('=======================')
        print('User config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, ':', getattr(self, k))
        print('=======================')
                
opt = DefaultConfig()
                

            