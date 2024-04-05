class CarFinder:
    def __init__(self, nuImage, min_size_x, min_size_y):
        self.nuim = nuImage
        self.min_size_x = min_size_x
        self.min_size_y = min_size_y

    def fetch_vehicles_bboxes_from_dataset(self):
        vehicles_bounding_boxes_dataset = {}
        for id, image in enumerate(self.nuim.sample):
            vehicles_bboxes = self.fetch_vehicles_bboxes_from_img(
                image
            )
            vehicles_bounding_boxes_dataset[id] = vehicles_bboxes
        return vehicles_bounding_boxes_dataset

    def fetch_vehicles_bboxes_from_img(self, image):
        vehicles_bounding_boxes = []
        objects_tokens, _ = self.nuim.list_anns(image["token"], verbose=False)

        for object_token in objects_tokens:
            object_data = self.nuim.get("object_ann", object_token)
            category = self.nuim.get("category", object_data["category_token"])["name"]

            if "vehicle" in category and self.check_bbox_size(object_data["bbox"]):
                vehicles_bounding_boxes.append(object_data["bbox"])
        return vehicles_bounding_boxes

    def check_bbox_size(self, bbox):
        x1, y1, x2, y2 = bbox
        return abs(x2-x1) > self.min_size_x and abs(y2-y1) > self.min_size_y
