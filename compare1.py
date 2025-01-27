import cv2
import face_recognition
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')  # Nano model for faster inference

# Load reference image and encode the face
reference_image = face_recognition.load_image_file("avatar (1).jpg")
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
reference_encodings = face_recognition.face_encodings(reference_image)

if len(reference_encodings) == 0:
    print("Error: No face found in the reference image.")
    exit()

reference_encoding = reference_encodings[0]  # Use the first face encoding

# Open webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video feed.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Exiting...")
        break

    # Convert frame to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform YOLO inference
    results = model(frame, verbose=False)

    # Iterate over YOLO detections
    for detection in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Convert to integers
        cropped_object = frame[y1:y2, x1:x2]  # Crop the detected object

        # Convert cropped object to RGB
        cropped_object_rgb = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)

        # Check if the cropped object is a face
        face_locations = face_recognition.face_locations(cropped_object_rgb)
        face_encodings = face_recognition.face_encodings(cropped_object_rgb, face_locations)

        if face_encodings:  # If at least one face is detected
            for face_encoding in face_encodings:
                # Compare with the reference face encoding
                match_results = face_recognition.compare_faces([reference_encoding], face_encoding)
                face_distance = face_recognition.face_distance([reference_encoding], face_encoding)[0]

                # Annotate the frame based on the comparison
                label = f"Match: {1 - face_distance:.2f}"  # Higher score means better match
                color = (0, 255, 0) if match_results[0] else (0, 0, 255)  # Green if match, Red otherwise
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # If no face is detected in the cropped region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "No Face Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the live feed with annotations
    cv2.imshow("YOLO Live Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
