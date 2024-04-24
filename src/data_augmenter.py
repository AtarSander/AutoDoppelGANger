import cv2
import os
import numpy as np
from src.progress_bar import ProgressBar

class DataAugmenter:
    def __init__(self, directory):
        self.directory = directory
        self.progress_bar = ProgressBar()

    def augment_dataset(self):
        filenames = os.listdir(self.directory)
        self.progress_bar.start(len(filenames))
        for f_id, filename in enumerate(filenames):
            augmented_images = self.create_variations(filename)
            self.progress_bar.update(100, f_id)
            for new_filename, image in augmented_images.items():
                self.save_image(image, new_filename)
        self.progress_bar.finish()

    def create_variations(self, filename):
        augmented_images = {}
        image = cv2.imread(self.directory+"/"+filename)
        augmented_images[self.set_filename("flip", filename)] = self.filp_horizontally(image)
        augmented_images[self.set_filename("-rotate", filename)] = self.rotate_image(image, -5)
        augmented_images[self.set_filename("+rotate", filename)] = self.rotate_image(image, 5)
        augmented_images[self.set_filename("contrast", filename)] = self.adjust_contrast(image, 1.4)
        augmented_images[self.set_filename("brightness", filename)] = self.adjust_brightness(image, 0.2)
        augmented_images[self.set_filename("gaussian", filename)] = self.add_gaussian_noise(image, 0, 3)
        return augmented_images

    def filp_horizontally(self, image):
        image_flipped_h = cv2.flip(image, 1)
        return image_flipped_h

    def rotate_image(self, image, angle):
        height, width = image.shape[:2]
        rotation_matrx = cv2.getRotationMatrix2D((height/2, width/2), angle, 1)
        image_rotated = cv2.warpAffine(image, rotation_matrx, (width, height), flags=cv2.INTER_LINEAR)
        return image_rotated

    def adjust_contrast(self, image, contrast_factor):
        image_float = image.astype(np.float32) / 255.0
        adjusted_image = (image_float) * contrast_factor
        return self.reformat_to_int(adjusted_image)

    def adjust_brightness(self, image, brightness_factor):
        image_float = image.astype(np.float32) / 255.0
        adjusted_image = image_float + brightness_factor
        return self.reformat_to_int(adjusted_image)

    def add_gaussian_noise(self, image, mean, sigma, scale_factor = 0.02):
        random_noise = np.random.normal(mean, sigma, image.shape)
        image_float = image.astype(np.float32) /255.0
        adjusted_image = image_float + random_noise * scale_factor
        return self.reformat_to_int(adjusted_image)

    def reformat_to_int(self, adjusted_image):
        adjusted_image = np.clip(adjusted_image, 0, 1)
        adjusted_image = (adjusted_image * 255).astype(np.uint8)
        return adjusted_image

    def set_filename(self, type, org_filename):
        tmp = org_filename.rstrip(".jpg")

        if type=='flip':
            tmp+= 'f'
        elif type=='-rotate':
            tmp+='-r'
        elif type=='+rotate':
            tmp+='+r'
        elif type== 'contrast':
            tmp+='c'
        elif type=='brightness':
            tmp+='b'
        elif type=='gaussian':
            tmp+='g'

        filename = tmp+ '.jpg'
        return filename

    def save_image(self, image, filename):
        dir_path = self.directory + filename
        cv2.imwrite(dir_path, image)