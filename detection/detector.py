from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):

        results = self.model(
            frame,
            conf=0.35,
            verbose=False
        )

        return results[0]