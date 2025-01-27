import cv2
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Load YOLO model
model = YOLO('yolov8n.pt')  # Use a nano model for faster inference

# Load reference image
reference_image = cv2.imread("bharat1.jpg")
reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Open webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video feed.")
    exit()

# Get frame dimensions for VideoWriter
ret, frame = cap.read()
if not ret:
    print("Error: Could not read initial frame.")
    cap.release()
    exit()

frame_height, frame_width = frame.shape[:2]
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 20, (frame_width, frame_height))

def compare_images(image1, image2):
    """Compare two images using Structural Similarity Index (SSIM)."""
    # Resize images to the same size for comparison
    image1_resized = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    # Compute SSIM
    score, _ = ssim(image1_resized, image2, full=True)
    return score

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Exiting...")
        break

    # Perform YOLO inference
    results = model(frame, verbose=False)

    # Iterate over detections
    for detection in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Convert to integers
        cropped_object = frame[y1:y2, x1:x2]  # Crop the detected object

        # Convert the cropped object to grayscale
        cropped_object_gray = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2GRAY)

        # Compare the cropped object with the reference image
        similarity_score = compare_images(cropped_object_gray, reference_image_gray)
        
        # Annotate the frame with the similarity score
        label = f"Match: {similarity_score:.2f}"
        color = (0, 255, 0) if similarity_score > 0.8 else (0, 0, 255)  # Green if similar, Red otherwise
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the annotated frame to the video
    out.write(frame)

    # Display the live feed with annotations
    cv2.imshow("YOLO Live Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
