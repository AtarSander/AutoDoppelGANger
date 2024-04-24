from data_preprocessor import DataPreprocessor
from car_cutter import CarCutter
from nuimages import NuImages
from gan import GAN
import torch
import json

class AutoDoppelGANgerShell:
    def __init__(self):
        print("Welcome to AutoDoppelGANger. Type 'help' to list commands.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GAN(64, 64, 3, 100, device, '../logs')
        self.dataset = None

    def help(self):
        print("Available commands:")
        print("lddst - Load dataset")
        print("train - Start training the model")
        print("eval - Evaluate the model")
        print("exit - Exit the shell")

    def cut_nu_images(self, dataroot, out_path, version, min_size_x, min_size_y):
        print("Cutting out cars...")
        nuim = NuImages(dataroot=dataroot, version=version, verbose=False, lazy=True)
        cutter = CarCutter(nuim, min_size_x, min_size_y)
        cutter.cut_out_vehicles_from_dataset(dataroot, out_path+"car_cut/")
        print("Done.")

    def load_dataset(self, filepath, target_width=64, target_height=64, img_channels=3):
        print("Loading dataset...")
        preprocess = DataPreprocessor(target_width, target_height, img_channels)
        self.dataset = preprocess.load_dataset(filepath)
        print("Done.")

    def train(self, num_epochs, batch_size, learning_rate, save_weights, 
                         beta1=0.5, beta2=0.999, time_limit=1):
        print("Training model...")
        if not self.dataset:
            print("You must load dataset with 'lddst <filepath>' before training model.")
            return
        self.model.train(self.dataset, num_epochs, batch_size, learning_rate,
                         beta1, beta2, time_limit)
        if save_weights:
            self.model.save_models_weights("../models/checkpoints")
        print("Done.")


    def eval(self):
        print("Evaluating model...")
        # Add evaluation logic here

    def exit(self):
        print("Exiting...")
        return False

    def run(self):
        running = True
        while running:
            user_input = input("AutoDoppelGANger> ").strip()
            if user_input:
                parts = user_input.split()
                if parts[0] == 'help':
                    self.help()
                elif parts[0] == 'lddst':
                    self.load_dataset(parts[1])
                elif parts[0] == 'train':
                    with open(parts[1], 'r') as file_handle:
                        training_setup = json.load(file_handle)
                    self.train(training_setup["num_epochs"],
                               training_setup["batch_size"],
                               training_setup["learning_rate"],
                               training_setup["save_weights"])
                elif parts[0] == 'eval':
                    self.eval()
                elif parts[0] == 'exit':
                    running = self.exit()
                else:
                    print("Unknown command. Type 'help' for help.")
            else:
                continue

