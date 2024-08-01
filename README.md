# A first GAN | Short Intro
Generative adversarial networks are machine learning structures that can learn to imitate a given distribution of data.
GANs consist of two neural networks, one trained to generate data and the other trained to distinguish fake data from real data.
And this is how this project navigates itself. We build an NN that produces data samples (generator) that are then fed to another
NN that decides on whether the samples are the expected or not. The whole costruction is implemented with the Pytorch library.
