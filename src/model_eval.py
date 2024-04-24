import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.linalg import sqrtm

class ModelEval:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.inception_model = models.inception_v3(weights='Inception_V3_Weights.DEFAULT', aux_logits=True)
        self.device = device
        self.inception_model.to(device)
        self.inception_model.eval()

    def compute_inception_score(self, generated_images):
        preprocessed_images = self.preprocess_data(generated_images)
        conditional_probs = self.calculate_conditional_probs(preprocessed_images)
        inception_score = self.calculate_inception_score(conditional_probs)
        return inception_score

    def compute_FID(self, data_loader, generated_images):
        real_images = self.load_real_images(data_loader, generated_images.shape[0])
        real_images = self.preprocess_data(real_images)
        generated_images = self.preprocess_data(generated_images)
        real_features = self.extract_features(real_images)
        generated_features = self.extract_features(generated_images)
        mu_real, mu_generated = self.calculate_mean(real_features, generated_features)
        sigma_real, sigma_generated = self.calculate_covariance(real_features, generated_features)
        fid = self.calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)
        return fid

    def load_real_images(self, data_loader, size):
        real_images = []
        count = 0
        for batch, _ in data_loader:
            for real in batch:
                real_images.append(real)
                count+=1
                if count >= size:
                    break
            if count >= size:
                    break
        return np.array(real_images)

    def preprocess_data(self, generated_images):
        preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        generated_images = [torch.from_numpy(image) for image in generated_images]
        preprocessed_images = torch.stack([preprocess(image) for image in generated_images])
        return preprocessed_images

    def calculate_conditional_probs(self, preprocessed_images):
        conditional_probs = []
        with torch.no_grad():
            for i in range(0, preprocessed_images.size(0), self.batch_size):
                batch_images = preprocessed_images[i:i+self.batch_size].to(self.device)
                outputs = self.inception_model(batch_images)
            conditional_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        conditional_probs = np.array(conditional_probs)
        return conditional_probs

    def calculate_inception_score(self, conditional_probs):
        marginal_distribution = np.mean(conditional_probs, axis=0)
        kl_divergences = np.sum(conditional_probs * (np.log(conditional_probs) - np.log(marginal_distribution)), axis=1)
        inception_score = np.exp(np.mean(kl_divergences))
        return inception_score

    def extract_features(self, preprocessed_images):
        features = []
        with torch.no_grad():
            for i in range(0, preprocessed_images.size(0), self.batch_size):
                batch_images = preprocessed_images[i:i+self.batch_size].to(self.device)
                outputs = self.inception_model(batch_images)
                features.append(outputs.cpu().numpy())
        return np.concatenate(features, axis=0)

    def calculate_mean(self, real_features, generated_features):
        mu_real = np.mean(real_features, axis=0)
        mu_generated = np.mean(generated_features, axis=0)
        return mu_real, mu_generated

    def calculate_covariance(self, real_features, generated_features):
        sigma_real = np.cov(real_features, rowvar=False) + np.eye(real_features.shape[1]) * 1e-6
        sigma_generated = np.cov(generated_features, rowvar=False) + np.eye(generated_features.shape[1]) * 1e-6
        return sigma_real, sigma_generated

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        sqrt_term, _ = sqrtm(sigma1 @ sigma2, disp=False)
        if not np.isfinite(sqrt_term).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            sqrt_term = sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)[0]
        print(type(sqrt_term))
        a = np.linalg.norm(mu1 - mu2)
        b = np.trace(sigma1 + sigma2 - 2 * sqrt_term)
        print(type(sigma1), type(sigma2))
        fid = a+b
        return fid


