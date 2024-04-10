from src.gan import GAN
from src.car_cutter import CarCutter
from src.data_preprocessor import DataPreprocessor
from nuimages import NuImages
import torch


SRC_PATH = "/mnt/d/data/raw/"
VERSION = "v1.0-train"
OUT_PATH = "dataset/"

if __name__ == "__main__":
    nuim = NuImages(dataroot=SRC_PATH, version=VERSION, verbose=False, lazy=True)
    cutter = CarCutter(nuim, 150, 150)
    cutter.cut_out_vehicles_from_dataset(SRC_PATH, OUT_PATH+"images/")
    preprocess = DataPreprocessor(64, 64, 3)
    preprocess.resize_dataset(OUT_PATH+"images/")
    dataset = preprocess.load_dataset(OUT_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAN(64, 64, 3, 100, device)
    model.train(dataset, 50, 128, 3e-4)
    samples = model.generate_samples(8)
    model.plot_grid(samples, 2, 4)
    model.save_models_weigths("models/checkpoints")