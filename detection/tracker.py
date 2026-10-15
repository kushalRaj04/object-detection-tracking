from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:

    def __init__(self, model_path="yolo11n.pt"):

        # Load YOLO model
        self.model = YOLO(model_path)

        # Create Deep SORT tracker
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0
        )


    def track(self, frame):

        # Run YOLO detection
        results = self.model(
            frame,
            conf=0.35,
            verbose=False
        )

        result = results[0]

        # List of detections for Deep SORT
        detections = []


        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()


            for box, confidence, class_id in zip(
                boxes,
                confidences,
                classes
            ):

                x1, y1, x2, y2 = box

                width = x2 - x1
                height = y2 - y1

                # Deep SORT expects:
                # [x, y, width, height]
                bbox = [
                    x1,
                    y1,
                    width,
                    height
                ]

                detections.append(
                    (
                        bbox,
                        float(confidence),
                        int(class_id)
                    )
                )


        # Run Deep SORT
        tracks = self.tracker.update_tracks(
            detections,
            frame=frame
        )


        # Store tracking information
        tracked_objects = []


        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            bbox = track.to_ltrb()

            x1, y1, x2, y2 = map(
                int,
                bbox
            )

            class_id = track.get_det_class()

            if class_id is None:
                continue

            tracked_objects.append(
                {
                    "track_id": track_id,
                    "class_id": int(class_id),
                    "bbox": (x1, y1, x2, y2)
                }
            )


        return tracked_objects