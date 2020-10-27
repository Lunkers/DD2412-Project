Reimplmentation of [GLOW](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf), a flow-based generative model, in Pytorch.

To run quantitative results on Cifar with our settings:

    python train.py --batch_size 16 -L 4 -K 8 --lr 1e-4 --dataset CIFAR --generate_samples

To run quantitative results on MNIST with our settings:

    python train.py --batch_size 16 -L 4 -K 8 --lr 1e-4 --dataset MNIST --norm_method batchnorm --generate_samples
    
To randomly generate samples from the trained model run from respective dataset with our settings (note only do this after you've finished training the model):

    python sample_images.py --dataset CIFAR
    python sample_images.py --dataset --norm_method batchnorm MNIST
