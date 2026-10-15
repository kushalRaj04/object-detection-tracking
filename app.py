from flask import Flask, render_template, Response, jsonify
from detection.tracker import ObjectTracker
import cv2

app = Flask(__name__)

# Create Deep SORT object tracker
tracker = ObjectTracker()

# Open webcam
camera = cv2.VideoCapture(0)


# Store latest statistics
statistics = {
    "object_count": 0,
    "tracked_count": 0,
    "object_types": {}
}


def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Run YOLO + Deep SORT
        tracked_objects = tracker.track(frame)

        # Total tracked objects
        object_count = len(tracked_objects)
        tracked_count = len(tracked_objects)

        # Count objects by type
        object_types = {}


        # Process every tracked object
        for obj in tracked_objects:

            x1, y1, x2, y2 = obj["bbox"]

            track_id = obj["track_id"]

            class_id = obj["class_id"]

            # Get class name from YOLO
            class_name = tracker.model.names[class_id]


            # Count object type
            if class_name in object_types:

                object_types[class_name] += 1

            else:

                object_types[class_name] = 1


            # Label
            label = (
                f"{class_name} "
                f"ID:{track_id}"
            )


            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


        # Update statistics
        statistics["object_count"] = object_count

        statistics["tracked_count"] = tracked_count

        statistics["object_types"] = object_types


        # Display total objects
        cv2.putText(
            frame,
            f"Objects Tracked: {tracked_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )


        # Convert frame to JPEG
        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )


        if not ret:
            continue


        frame_bytes = buffer.tobytes()


        # Send frame to browser
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stats")
def stats():

    return jsonify(statistics)


if __name__ == "__main__":

    app.run(
        debug=True
    )