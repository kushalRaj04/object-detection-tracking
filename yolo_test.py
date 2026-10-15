from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("YOLO object detection started.")
print("Press Q to quit.")

while True:

    
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

   
    annotated_frame = results[0].plot()

    
    cv2.imshow("YOLO Object Detection", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()

# Close windows
cv2.destroyAllWindows()