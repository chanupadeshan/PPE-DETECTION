from ultralytics import YOLO
import torch






train_model = YOLO('../yolo weight/yolo11x.pt') 
train_model.to('cuda:0') # Force model to use GPU 

yaml_path = "/home/chanupa/Documents/projects/cv/PPE detection/Construction Site Safety/data.yaml"

train_model.train(
    data=yaml_path,            # Path to dataset YAML
    epochs=100,                 # Number of training epochs
    imgsz=416,                 # Input image size 
    batch=8,                   # Batch size per iteration
    device=0,                  # GPU device id
    workers=4,                 # Number of data loader workers
    patience=10,               # Early stopping patience
    optimizer="SGD",           # Try SGD or AdamW (default is auto)
    cos_lr=True,               # Use cosine learning rate scheduler
    amp=True,                  # Mixed precision (faster on supported GPUs)
    verbose=True               # Print more training info
)