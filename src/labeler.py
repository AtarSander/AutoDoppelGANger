import os
import cv2
from natsort import natsorted
import matplotlib.pyplot as plt

class Labeler:
    def label_loop(self, directory, sort_direction=None):
        filenames = self.sort_type(directory, sort_direction)

        for filename in filenames:
            filepath = os.path.join(directory, filename)
            print(filename)
            self.print_image(filepath, filename)
            self.label_file(filename, filepath)

    def label_file(self, filename, filepath):
        option = input("f-front, b-back, s-side, d-discard: ")

        if option.strip() == "f":
            os.rename(filepath, "datasets_labeled/front/images/"+filename)
        elif option.strip() == "b":
            os.rename(filepath, "datasets_labeled/back/images/"+filename)
        elif option.strip() == "s":
            os.rename(filepath, "datasets_labeled/side/images/"+filename)
        else:
            os.remove(filepath)

    def sort_type(self, directory, sort_direction):
        if sort_direction is None:
            return natsorted(os.listdir(directory))
        else:
            return natsorted(os.listdir(directory), reverse=True)

    def print_image(self, filepath, filename):
        image = cv2.imread(filepath)
        cv2.imshow(filename, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




labeler = Labeler()
labeler.label_loop("dataset/images", sort_direction=1)