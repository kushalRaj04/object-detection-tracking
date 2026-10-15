from detection.tracker import ObjectTracker
import cv2


tracker = ObjectTracker()

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("Could not open webcam")
    exit()


print("Deep SORT tracking started.")
print("Press Q to quit.")


while True:

    success, frame = camera.read()

    if not success:
        break


    tracked_objects = tracker.track(frame)


    for obj in tracked_objects:

        x1, y1, x2, y2 = obj["bbox"]

        track_id = obj["track_id"]

        class_id = obj["class_id"]

        class_name = tracker.model.names[class_id]


        label = (
            f"{class_name} "
            f"ID:{track_id}"
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "Deep SORT Object Tracking",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()

cv2.destroyAllWindows()