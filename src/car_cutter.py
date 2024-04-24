from src.car_finder import CarFinder
from src.progress_bar import ProgressBar
import cv2


class CarCutter:
    def __init__(self, nuImages, min_size_x, min_size_y):
        self.nuImages = nuImages
        self.carFinder = CarFinder(nuImages, min_size_x, min_size_y)
        self.progress_bar = ProgressBar()

    def cut_out_vehicles_from_dataset(self, src_dir, out_dir):
        dataset_size = len(self.nuImages.sample)
        self.progress_bar.start(dataset_size)

        for img_id, image in enumerate(self.nuImages.sample):
            vehicles_bboxes = self.carFinder.fetch_vehicles_bboxes_from_img(
                image
            )
            sample_data = self.nuImages.get("sample_data", image["key_camera_token"])
            src_path = src_dir+sample_data["filename"]
            out_path = out_dir+"image"+str(img_id)+"_car"
            self.progress_bar.update(100, img_id)
            for car_id, vehicle_bbox in enumerate(vehicles_bboxes):
                self.cut_out_bbox(src_path, vehicle_bbox, out_path+str(car_id)+".jpg")
        self.progress_bar.finish()

    def cut_out_bbox(self, image_path, bbox, output_path):
        image = cv2.imread(image_path)
        x1, y1, x2, y2 = bbox
        bbox_image = image[y1:y2, x1:x2]
        cv2.imwrite(output_path, bbox_image)
