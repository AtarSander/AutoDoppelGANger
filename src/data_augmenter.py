import cv2
import numpy as np

class DataAugmenter:
    def create_variations(self, image_path):
        image = cv2.imread(image_path)

        mean_pixel_value = np.mean(image)
        adjusted_image = (image - mean_pixel_value) * contrast_factor + mean_pixel_value

    def filp_horizontally(self, image):
        image_flipped_h = cv2.flip(image, 1)
        return image_flipped_h

    def rotate_image(self, image, angle):
        height, width = image.shape[:2]
        rotation_matrx = cv2.getRotationMatrix2D((height/2, width/2), angle, 1)
        image_rotated = cv2.warpAffine(image, rotation_matrx, (width, height), flags=cv2.INTER_LINEAR)
        return image_rotated

    def adjust_contrast(self, image, contrast_factor):
        mean_pixel_value = np.mean(image)
        image_float = image.astype(np.float32) / 255.0
        adjusted_image = (image_float - mean_pixel_value) * contrast_factor
        adjusted_image = np.clip(adjusted_image, 0, 1)
        adjusted_image = (adjusted_image * 255).astype(np.uint8)
        return adjusted_image

    def adjust_brightness(self, image, brightness_factor):
        image_float = image.astype(np.float32) / 255.0
        adjusted_image = image_float + brightness_factor
        adjusted_image = np.clip(adjusted_image, 0, 1)
        adjusted_image = (adjusted_image * 255).astype(np.uint8)
        return adjusted_image