# A First GAN | An Intro to GenAI
Generative adversarial networks are machine learning structures that can learn to imitate a given distribution of data.
GANs consist of two neural networks, one trained to generate data and the other trained to distinguish fake data from real data.
And this is how this project navigates itself. We build an NN that produces data samples (generator) that are then fed to another
NN (discriminator) that decides on whether the samples are the expected ones or not. The whole costruction is implemented with the Pytorch library.
This project initiates an exploration in the world of GenAI, as similar NN structure will be used in LLM-inspired projects in the future.


### The Project

GAN: The Discriminator - Generator Interplay. (Unofficial Notes: The Generator is feeding the Discriminator which then performs
a binary classification, Generator outputs a 2-dim tuple for each 2-dim input tuple)

```{p}
import torch
from torch import nn
import math
import matplotlib.pyplot as plt
```

<br/>
For the purpose of reproducibility

```{p}
torch.manual_seed(111)
```
<br/>
Creating the training set. It composes from the tuples (six, cosx) for x in [0,2*pi]. As we are in an unsupervided framework
a label-classification class column is created, consisted of zeroes only. 

```{p}
train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
#train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
#train_data[:, 1] = torch.sin(train_data[:, 0])
int =  torch.rand(train_data_length)
train_data[:, 0] = torch.sin(2 * math.pi * int) 
train_data[:, 1] = torch.cos(2 * math.pi * int)
train_labels = torch.zeros(train_data_length)
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]
```
<br/>

Lets plot the graph we want our NN to reconstruct.

```{p}
plt.plot(train_data[:, 0], train_data[:, 1], ".")
plt.title("Train Set")
plt.xlabel("sinx(x)")
plt.ylabel("cos(x)")
```
<br/>

![download](https://github.com/user-attachments/assets/e2e5bd88-3bd9-4ff5-84fc-2d58ecbc9a9d)

<br/>
We now split the train set randomly into 32 batches and we then create the required object for training the NN's below.

```{p}
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
```
<br/>

The first NN(in fact,a CNN, MLP as well): A 2-dimensional input layer, three hidden layers of 256,128 and 64 neurons each, and an one-dimensional output layer  
Rectified Linear Unit (RELU) is the activation function used in all hidden layers, while the Sigmoid is employed for the last layer, providing a 
a probabilistic output.
Finally we define a function that will run the NN.

```{p}
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

discriminator = Discriminator()
```



The second NN(in fact,a CNN, MLP as well): A 2-dimensional input layer, two hidden layers of 16 and 32 neurons each, and an two-dimensional output layer  
Again, Rectified Linear Unit (RELU) is the activation function used in all hidden layers.
Finally we define a function that will call and run the NN from the Generator class.

```{p}
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()
```
<br/>
Optimisation Algorithm and Loss Function: The update of the parameters (weights and biases) for each neuron and layer will materialise
through the stochastic gradient descent of the Binary Cross Entropy function. We set the learning rate for the SGD to 0.001 while also
the number of training iterations(epochs) is provided.

```{p}
lr = 0.001#Learning Rate
num_epochs = 300
loss_function = nn.BCELoss()
```
<br/>
Here, we employ the Adam Optimization Algorithm.
```{p}
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
```
<br/>


Now, we procced into training our model. Details are provided throughout. Then the results of the loss function for each NN are provided.

```{p}
for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data preprocessing and labelling
        real_samples_labels = torch.ones((batch_size, 1))# We set real data labels to 1, using these labels will be then classified by the discriminator
        latent_space_samples = torch.randn((batch_size, 2))# Random input for the Generative NN.
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))# We set the generated samples label to 0.
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator first using Backpropagation
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)# The loss function uses the probabilities assigned to each entry of the batch 
                                                     #by the discriminator and the (artificially made) labels 1 to the real values, 0 to the random ones.)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 2))

        # Training the generator.
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )# Here, in contrast to the training of the discriminator, we only use the real sample labels for computing the loss function.
         # In other words, we assign 1 to all the (already trained)probabilities given on the random input(latent space) and we check the 
         # the performance of the loss function. 
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
```
![Screenshot 2024-08-22 123814](https://github.com/user-attachments/assets/b0a5e2dc-ba9a-4491-8f03-44a1ee336fb0)

<br/>

Lets now examine visually how training affects the model performance. Without Training:

```{p}
latent_space_samples = torch.randn((batch_size, 2))# Random input for the Generative NN.
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:,0], generated_samples[:,1], ".")
```
![download](https://github.com/user-attachments/assets/c7bb2b09-db47-4d97-a086-c20b0a27245f)

After training for 30 epochs with the (real) input data being elements of the unit circle.
```{p}
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:,0], generated_samples[:,1], ".")
```
![download](https://github.com/user-attachments/assets/aaca7fb7-d29b-4fca-9031-0a3e51144223)

After training for 50 epochs with the (real) input data being elements of the unit circle. There are still few points misaligned.

```{p}
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:,0], generated_samples[:,1], ".")
```
![download](https://github.com/user-attachments/assets/1544eea4-cd1b-44fc-ab63-4af6a6442a69)


After training the GAN model for 300 epochs with the (real) input data being elements of the unit circle.
Observe that there is only one point standing out of the circle figure, which we can guarantee that is will
also be classified correctly by the model at the event of higher-epoch training. 

```{p}
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:,0], generated_samples[:,1], ".")
```
![download](https://github.com/user-attachments/assets/7696ad71-8dad-4dbc-8d4e-d677a43e1e10)



