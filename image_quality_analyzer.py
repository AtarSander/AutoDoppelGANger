import cv2
import warnings
import os
import numpy as np
import pandas as pd


class ImageQualityAnalyzer:
    def set_image(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def analyze_dataset(self, src_dir):
        df = pd.DataFrame(columns=["Image", "Brightness", "Sharpness", "Focus",
                                   "Contrast", "Noise Frequency", "Color Balance Red",
                                   "Color Balance Green", "Color Balance Blue",
                                   "Uniformity"])
        for filename in os.listdir(src_dir):
            # pandas 2.2.1 has a FutureWarning for concatenating DataFrames with Null entries
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.set_image(src_dir+"/"+filename)
                df = pd.concat([df, pd.DataFrame(self.analyze_image(), index=[0])],
                               ignore_index=True)

    def analyze_image(self):
        image_metrics = {}
        image_metrics["Image"] = self.image_path
        image_metrics["Brightness"] = self.brightness()
        image_metrics["Sharpness"] = self.sharpness()
        image_metrics["Focus"] = self.focus()
        image_metrics["Contrast"] = self.contrast()
        image_metrics["Noise Frequency"] = self.noise_frequency()
        balance_red, balance_green, balance_blue = self.color_balance()
        image_metrics["Color Balance Red"] = balance_red
        image_metrics["Color Balance Green"] = balance_green
        image_metrics["Color Balance Blue"] = balance_blue
        image_metrics["Uniformity"] = self.uniformity()
        return image_metrics

    def brightness(self):
        avg_brightness = cv2.mean(self.gray_image)[0]

        return avg_brightness

    def sharpness(self):
        laplacian = cv2.Laplacian(self.gray_image, cv2.CV_64F)
        sharpness = laplacian.var()

        return sharpness

    def focus(self):
        grad_x = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient_magnitude = np.mean(gradient_magnitude)

        return avg_gradient_magnitude

    def contrast(self):
        contrast = np.std(self.gray_image)

        return contrast

    def noise_frequency(self, threshold=0.1):
        f_transform = np.fft.fft2(self.image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
        noise_mask = magnitude_spectrum > threshold * np.max(magnitude_spectrum)
        noisy_pixels = np.sum(noise_mask)

        total_pixels = self.image.shape[0] * self.image.shape[1]
        noise_percentage = (noisy_pixels / total_pixels) * 100

        return noise_percentage

    def color_balance(self):
        hist_red = cv2.calcHist([self.image], [2], None, [256], [0, 256])
        hist_green = cv2.calcHist([self.image], [1], None, [256], [0, 256])
        hist_blue = cv2.calcHist([self.image], [0], None, [256], [0, 256])

        hist_red /= np.sum(hist_red)
        hist_green /= np.sum(hist_green)
        hist_blue /= np.sum(hist_blue)

        variance_red = np.var(hist_red)
        variance_green = np.var(hist_green)
        variance_blue = np.var(hist_blue)

        return (variance_red, variance_green, variance_blue)

    def uniformity(self):
        window_size = 5
        local_variances = cv2.blur(self.image, (window_size, window_size))

        local_variances = local_variances / 255.0

        avg_variance = np.mean(local_variances)

        return avg_variance
