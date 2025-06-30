# PPE Detection with YOLO

This project detects Personal Protective Equipment (PPE) in videos using a YOLO model and OpenCV.

## Dataset

- [Construction Site Safety Image Dataset (Roboflow) on Kaggle](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)

Download the dataset from the above link and prepare your training data as needed.

## Requirements

- Python 3.8+
- [ultralytics](https://pypi.org/project/ultralytics/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [cvzone](https://pypi.org/project/cvzone/)
- [numpy](https://pypi.org/project/numpy/)

Install dependencies with:
```bash
pip install ultralytics opencv-python cvzone numpy
```

## Usage

1. Download and extract the dataset from the Kaggle link above.
2. Place your video file in the `video/` directory (e.g., `video/ppe-3-1.mp4`).
3. Make sure your trained YOLO model weights are in the correct path (e.g., `runs/detect/train2/weights/best.pt`).
4. Edit `PPE.py` if you want to change the video or model path.
5. Run the detection script:
```bash
python PPE.py
```

- The script will display the video with bounding boxes and labels for detected PPE items.
- Press `q` to quit the video window.

## Notes
- If you see errors about missing modules, ensure you have activated the correct Python environment and installed all requirements.
- If you see errors about missing files, check that the video and model paths in `PPE.py` are correct.

## Project Structure
```
PPE detection/
├── PPE.py
├── video/
│   └── ppe-3-1.mp4
├── runs/
│   └── detect/
│       └── train2/
│           └── weights/
│               └── best.pt
├── README.md
```


test test