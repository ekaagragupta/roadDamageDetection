# Usage Guide

## Training
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0  # GPU
)
```

## Inference
```python
# Load trained model
model = YOLO('weights/best.pt')

# Predict on image
results = model.predict('test_image.jpg', conf=0.25)

# Display results
results[0].show()
```

## Complete Pipeline

See [notebooks/detect.ipynb](../notebooks/detect.ipynb) for:
- Detection
- Severity scoring
- Cost estimation
- GPS tagging
- Prioritization
- Clustering

## API (Coming Soon)

RESTful API with FastAPI for production deployment.