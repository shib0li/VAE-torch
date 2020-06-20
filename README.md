# Variational Auto-encoder with PyTorch

This is a light implementation of the Variational Auto-encoder(VAE) with [_PyTorch_](https://pytorch.org/) and tested on [_MNIST_](http://yann.lecun.com/exdb/mnist/) dataset. 

## System Requirement

The code is tested with python 3.7.7 on Ubuntu 18.04. The torch version installed is 1.3.1. 

## Run

See the Demo.ipynb to find the running configuration options in details.

## Experiment 1

The first experiment conducted is to test the images reconstruction. A batch of 64 images are drawn from the testing dataset, first pass to the encoder to acquire their latent encodings, then pass to the decoder to see if the VAE could recover the original images properly.

![ground_truth](https://raw.githubusercontent.com/shib0li/Scalable-GPRN/master/figures/ground.png)
