import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torchvision
from src.model import Generator, Discriminator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class GAN:
    def __init__(self, features_g, features_d, channels_img, noise_dim, device, log_dir):
        self.generator = Generator(noise_dim, channels_img, features_g, device)
        self.discriminator = Discriminator(channels_img, features_d, device)
        self.device = device
        self.noise_dim = noise_dim
        self.log_dir = log_dir

    def train(self, dataset, num_epochs, batch_size, learning_rate, beta1=0.5, beta2=0.999, time_limit = 1):
        self.load_data(dataset, batch_size)
        self.initialize_weigths()
        self.setup_optimizers(learning_rate, beta1, beta2)
        criterion = nn.BCELoss()

        self.generator.train()
        self.discriminator.train()

        fixed_noise = self.setup_tensorboard()
        step = 0

        start_time = time.time()
        epoch = 0
        while ((time.time() - start_time)/60 < time_limit):
        # for epoch in range(num_epochs):
            for batch_idx, (real, _) in enumerate(self.loaded_data):
                real = real.to(self.device)
                fake = self.generate_fake_input(batch_size)

                loss_dsc = self.loss_discriminator(criterion, real, fake)
                self.backprop_discriminator(loss_dsc)

                loss_gen = self.loss_generator(criterion, fake)
                self.backprop_generator(loss_gen)

                if batch_idx % 100 == 0:
                    self.print_training_stats(num_epochs, epoch, batch_idx,
                                              len(self.loaded_data), loss_gen, loss_dsc)
                    # generated_images = self.generate_samples(10, fixed_noise)
                    # self.plot_grid(generated_images, 5, 2, index=batch_idx)
                    self.tensor_board_grid(fixed_noise, self.writer_real, self.writer_fake, real, step)
                    step += 1
            epoch+=1
            self.print_time(start_time)

    def load_data(self, dataset, batch_size):
        self.loaded_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def initialize_weigths(self):
        self.generator.initialize_weights()
        self.discriminator.initialize_weights()

    def load_weights(self, gen_filepath, dsc_filepath):
        pass

    def setup_optimizers(self, learning_rate, beta1, beta2):
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.opt_dsc = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    def generate_fake_input(self, batch_size):
        noise = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
        return self.generator(noise)

    def loss_discriminator(self, criterion, real, fake):
        dsc_real = self.discriminator.forward(real).reshape(-1)
        dsc_fake = self.discriminator.forward(fake).reshape(-1)
        loss_dsc_real = criterion(dsc_real, torch.ones_like(dsc_real))
        loss_dsc_fake = criterion(dsc_fake, torch.zeros_like(dsc_fake))
        return (loss_dsc_real + loss_dsc_fake) * 0.5

    def backprop_discriminator(self, loss_dsc):
        self.opt_dsc.zero_grad()
        loss_dsc.backward(retain_graph=True)
        self.opt_dsc.step()

    def loss_generator(self, criterion, fake):
        gen_fake = self.discriminator.forward(fake).reshape(-1)
        return criterion(gen_fake, torch.ones_like(gen_fake))

    def backprop_generator(self, loss_gen):
        self.opt_gen.zero_grad()
        loss_gen.backward()
        self.opt_gen.step()

    def print_training_stats(self, num_epochs, epoch, batch_idx, dataset_size, loss_gen, loss_dsc):
        print(
              f"EPOCH: [{(epoch+1):2d}/{num_epochs:2d}], Batch [{batch_idx:3d} / {dataset_size:3d}] \
                   Loss Generator: {loss_gen:.6f}, Loss Discriminator: {loss_dsc:.6f}"
             )

    def print_time(self, start_time):
        print(f"Time elapsed: {((time.time() - start_time)/60):.2f} min")

    def generate_samples(self, num_samples, noise=None):
        if noise is None:
            noise = torch.randn(num_samples, self.noise_dim, 1, 1).to(self.device)
        with torch.no_grad():
            generated_samples = self.generator.forward(noise)
        return generated_samples.cpu()

    def setup_tensorboard(self):
        self.writer_real = SummaryWriter(self.log_dir+"/real")
        self.writer_fake = SummaryWriter(self.log_dir+"/fake")
        fixed_noise = torch.randn(32, self.noise_dim, 1, 1).to(self.device)
        return fixed_noise

    def save_models_weigths(self, path_dsc, path_gen):
        torch.save(self.discriminator.state_dict(), path_dsc)
        torch.save(self.generator.state_dict(), path_gen)

    def load_model_weights(self, path_dsc, path_gen):
        self.discriminator.load_state_dict(torch.load(path_dsc))
        self.generator.load_state_dict(torch.load(path_gen))

    # TODO make visualization into separate class
    def plot_grid(self, images, num_rows, num_cols, figsize=(10, 10), index=0):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        # images = images.detach().numpy()
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].permute(1, 2, 0).clamp(0, 1))
            ax.axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig("wykres"+str(index)+".png")

    def tensor_board_grid(self, fixed_noise, writer_real, writer_fake, real, step):
        with torch.no_grad():
            fake = self.generator.forward(fixed_noise)
            img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

            writer_real.add_image("Real", img_grid_real, global_step=step)
            writer_fake.add_image("Fake", img_grid_fake, global_step=step)
