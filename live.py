import cv2
import face_recognition
from ultralytics import YOLO
from datetime import datetime, time
import time as timer

# Load YOLO model
model = YOLO('yolov8n.pt')  # Nano model for faster inference

# Function to take a reference photo and encode the face
def capture_reference_photo():
    print("Press 's' to capture your photo for verification.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture photo.")
            exit()

        cv2.imshow("Capture Reference Photo", frame)

        # Wait for 's' to save the photo
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("reference_photo.jpg", frame)
            print("Photo captured and saved as reference_photo.jpg.")
            break

    cv2.destroyWindow("Capture Reference Photo")
    return frame

# Function to encode a face from a given image
def encode_face(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    encodings = face_recognition.face_encodings(rgb_image)
    if len(encodings) == 0:
        print("Error: No face found in the image.")
        return None
    return encodings[0]  # Return the first face encoding

# Open webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video feed.")
    exit()

# Step 1: Capture reference photo and encode it
reference_frame = capture_reference_photo()
reference_encoding = encode_face(reference_frame)
if reference_encoding is None:
    print("Exiting: No face detected in the reference photo.")
    cap.release()
    exit()

print("Reference face encoding captured successfully.")

# Step 2: Initialize variables for absence detection
absence_timer_start = None
max_absence_duration = 10  # Max allowed absence duration in seconds
attendance_log = []
student_present = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Exiting...")
        break

    # Convert frame to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform YOLO inference
    results = model(frame, verbose=False)

    student_found = False  # Flag to check if the student is detected

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

                match_confidence = 1 - face_distance
                label = f"Match: {match_confidence:.2f}"
                color = (0, 255, 0) if match_results[0] else (0, 0, 255)  # Green if match, Red otherwise

                if match_results[0] and match_confidence > 0.6:  # If the student matches the reference
                    student_found = True
                    student_present = True
                    absence_timer_start = None  # Reset the absence timer
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if timestamp not in attendance_log:
                        attendance_log.append(timestamp)
                        print(f"Verified at {timestamp}")
                    label = f"Verified: {match_confidence:.2f}"
                else:
                    label = "Not Verified"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if not student_found:
        if absence_timer_start is None:  # Start the absence timer if the student is not found
            absence_timer_start = timer.time()
        elif timer.time() - absence_timer_start > max_absence_duration:  # Check if absence exceeds the limit
            print("Student not on screen for 10 seconds. Exiting...")
            with open("attendance_log.txt", "w") as log_file:
                log_file.write("Attendance Log\n")
                log_file.write("===================\n")
                for log in attendance_log:
                    log_file.write(f"{log}\n")
                log_file.write("Status: Student not present for the exam.\n")
            break
    else:
        absence_timer_start = None  # Reset timer if the student is detected

    # Display the live feed with annotations
    cv2.imshow("Examination Verification System", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save attendance log if exiting
if not student_present:
    with open("attendance_log.txt", "w") as log_file:
        log_file.write("Attendance Log\n")
        log_file.write("===================\n")
        for log in attendance_log:
            log_file.write(f"{log}\n")
        log_file.write("Status: Student not present for the exam.\n")

print("Attendance log saved as attendance_log.txt.")

# Release resources
cap.release()
cv2.destroyAllWindows()