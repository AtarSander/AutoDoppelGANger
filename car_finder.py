class CarFinder:
    def __init__(self, nuImage):
        self.nuim = nuImage

    def fetch_vehicles_bboxes_from_dataset(self):
        vehicles_positions_for_image = {}
        for id, image in enumerate(self.nuim.sample):
            vehicles_bboxes = self.fetch_vehicles_bboxes_from_img(
                image
            )
            vehicles_positions_for_image[id] = vehicles_bboxes
        return vehicles_positions_for_image

    def fetch_vehicles_bboxes_from_img(self, image):
        vehicles_bounding_boxes = []
        objects_tokens, _ = self.nuim.list_anns(image["token"], verbose=False)

        for object_token in objects_tokens:
            object_data = self.nuim.get("object_ann", object_token)
            category = self.nuim.get("category", object_data["category_token"])["name"]

            if "vehicle" in category:
                vehicles_bounding_boxes.append(object_data["bbox"])
        return vehicles_bounding_boxes
