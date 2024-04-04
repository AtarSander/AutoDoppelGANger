import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from torch.utils.data import DataLoader


class GAN:
    def __init__(self, features_g, features_d, channels_img, noise_dim, device):
        self.generator = Generator(noise_dim, channels_img, features_g)
        self.discriminator = Discriminator(channels_img, features_d)
        self.device = device
        self.noise_dim = noise_dim

    def train(self, dataset, num_epochs, batch_size, learning_rate, beta1=0.9, beta2=0.999):
        loaded_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.initialize_weigths()
        self.setup_optimizers(learning_rate, beta1, beta2)
        criterion = nn.BCELoss()

        self.generator.train()
        self.discriminator.train()

        for epoch in range(num_epochs):
            for batch_idx, (real, _) in enumerate(loaded_data):
                real = real.to(self.device)
                fake = self.generate_fake_input(batch_size)

                loss_dsc = self.loss_discriminator(criterion, real, fake)
                self.backprop_discriminator(loss_dsc)

                loss_gen = self.loss_generator(criterion, fake)
                self.backprop_generator(loss_gen)

            if batch_idx % 100 == 0:
                self.print_training_stats(num_epochs, epoch, batch_idx,
                                          len(loaded_data), loss_gen, loss_dsc)
                generated_images = self.generate_samples(10)
                self.plot_grid(generated_images, 5, 2)
                
    def initialize_weigths(self):
        self.generator.initialize_weights()
        self.discriminator.initialize_weights()

    def setup_optimizers(self, learning_rate, beta1, beta2):
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.opt_dsc = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    def generate_fake_input(self, batch_size):
        noise = torch.randn((batch_size, self.noise_dim, 1, 1)).to(self.device)
        return self.generator(noise)

    def loss_discriminator(self, criterion, real, fake):
        dsc_real = self.discriminator(real).reshape(-1)
        dsc_fake = self.discriminator(fake).reshape(-1)
        loss_dsc_real = criterion(dsc_real, torch.ones_like(dsc_real))
        loss_dsc_fake = criterion(dsc_fake, torch.zeros_like(dsc_fake))
        return loss_dsc_real + loss_dsc_fake
    
    def backprop_discriminator(self, loss_dsc):
        self.opt_dsc.zero_grad()
        loss_dsc.backward(retain_graph=True)
        self.opt_dsc.step()

    def loss_generator(self, criterion, fake):
        gen_fake = self.discriminator(fake).reshape(-1)
        return criterion(gen_fake, torch.ones_like(gen_fake))
    
    def backprop_generator(self, loss_gen):
        self.opt_gen.zero_grad()
        loss_gen.backward()
        self.opt_gen.step()

    def print_training_stats(self, num_epochs, epoch, batch_idx, dataset_size, loss_gen, loss_dsc):
        print(
                    f"EPOCH: [{epoch}/{num_epochs}], Batch [{batch_idx} / {dataset_size}]\
                        Loss Generator: {loss_gen}, Loss Discriminator: {loss_dsc}"
             )
        
    def generate_samples(self, num_samples):
        noise = torch.randn((num_samples, self.noise_dim, 1, 1)).to(self.device)
        with torch.no_grad():
            generated_samples = self.generator(noise)
        return generated_samples
    
    # TODO make visualization into separate class
    def plot_grid(self, images, num_rows, num_cols, figsize=(10, 10)):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].permute(1, 2, 0).clamp(0, 1))  
            ax.axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  
        plt.show()
    