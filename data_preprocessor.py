import cv2


class DataPreprocessor:
    def __init__(self, target_width, target_height):
        self.target_width = target_width
        self.target_height = target_height

    def resize_image(self, image_path):
        image = cv2.imread(image_path)
        if self.is_too_small(image):
            resized_img = cv2.resize(image, (self.target_width, self.target_height),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = cv2.resize(image, (self.target_width, self.target_height),
                                     interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, resized_img)

    def is_too_small(self, image):
        height, width, channels = image.shape
        return height < self.target_height and width < self.target_width
