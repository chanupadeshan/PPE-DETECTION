from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import os

# Check if video file exists
video_path = "video/ppe-3-1.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found at {video_path}")

# Initialize video capture
capture = cv2.VideoCapture(video_path)
if not capture.isOpened():
    raise RuntimeError("Failed to open video capture")

# Check if model file exists
model_path = "runs/detect/train2/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the model
model = YOLO(model_path)

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck', 'truck and trailer', 'van', 'vehicle', 'wheel loader']

# Define colors for visualization
COLORS = {
    'box': (0, 255, 0),  # Green for boxes
    'text': (255, 0, 255),  # White for text
    'background': (0, 0, 0)  # Black for text background
}

try:
    while True:
        success, img = capture.read()
        
        if not success:
            print("End of video or failed to read frame")
            break
            
        img = cv2.resize(img, (1280, 720))
        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                clsIndex = int(box.cls[0])
                conf = box.conf[0]
                conf = int(conf * 100)
                if conf > 50:
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), COLORS['box'], 2)
                    
                    # Prepare text
                    label = f'{classNames[clsIndex]} {conf}%'

                    # Draw text
                    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
            
        cv2.imshow("PPE Detection", img)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    # Clean up
    capture.release()
    cv2.destroyAllWindows()

