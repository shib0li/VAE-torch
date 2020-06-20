# Variational Auto-encoder with PyTorch

This is a light implementation of the [_Variational Auto-encoder(VAE)_](https://arxiv.org/abs/1312.6114) with [_PyTorch_](https://pytorch.org/) and tested on [_MNIST_](http://yann.lecun.com/exdb/mnist/) dataset. 

## System Requirement

The code is tested with python 3.7.7 on Ubuntu 18.04. The torch version installed is 1.3.1. 

## Run

See the Demo.ipynb to find the running configuration options in details.

![loss](https://github.com/shib0li/VAE-torch/blob/master/figures/loss_hist.png)

## Experiment 1

The first experiment conducted is to test the images reconstruction. A batch of 64 images are drawn from the testing dataset, first pass to the encoder to acquire their latent encodings, then pass to the decoder to see if the VAE could recover the original images properly.

#### ground truth of testing
![ground_truth](https://github.com/shib0li/VAE-torch/blob/master/figures/ground.png)

#### reconstruction from decoder of testing
![recover](https://github.com/shib0li/VAE-torch/blob/master/figures/recover.png)

## Experiment 2

Generate artificial images from standard Gaussian noise. We can see the 'fake' generated images are reasonable. This reveals an important property of VAE which is distribution transformation. VAE transform from a simple (standard Gaussian) distribution to a very complicated distribution exsits in MNIST. 

![noise](https://github.com/shib0li/VAE-torch/blob/master/figures/noise.png)

## Reference

* [_Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013)._](https://arxiv.org/abs/1312.6114)
* [_Doersch, Carl. "Tutorial on variational autoencoders." arXiv preprint arXiv:1606.05908 (2016)._](https://arxiv.org/abs/1606.05908)


