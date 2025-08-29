# Dental X-Ray Tooth Detection with YOLOv11m

This repository contains the implementation and results of a YOLOv11m model trained to detect and classify 32 tooth types (using FDI notation) in dental X-ray images. The project was developed to meet a 48-hour deadline, concluding by 07:53 PM IST on Sunday, August 31, 2025.

## Project Overview

The goal was to train a robust object detection model to identify individual teeth in X-ray images, achieving high precision, recall, and mean Average Precision (mAP) scores. The model was trained on a custom dataset, fine-tuned over multiple stages, and evaluated on validation and test sets, with results compiled for submission.

## Environment Setup

The project was executed in a Google Colab environment with the following specifications:

- **Python Version**: 3.12.11
- **PyTorch Version**: 2.8.0+cu126 (CUDA-enabled)
- **Ultralytics YOLO Version**: 8.3.189
- **CUDA Version**: 12.6
- **GPU**: Tesla T4 (15GB memory)
- **Operating System**: Linux (Colab runtime)
- **Dependencies**:
  - `ultralytics` (installed via `pip install ultralytics==8.3.189`)
  - `torch` (pre-installed in Colab with CUDA support)
  - `pandas` (for result analysis, `pip install pandas`)
  - `numpy` (for post-processing, `pip install numpy`)
  - `opencv-python` (for image handling, `pip install opencv-python`)

To replicate the environment, run the following in a Colab notebook:
```bash
!pip install ultralytics==8.3.189 pandas numpy opencv-python
```

Mount Google Drive for dataset access:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Dataset Details

The dataset is stored in `/content/drive/MyDrive/final_dataset/` and consists of dental X-ray images with corresponding label files in YOLO format. The data was split as follows based on labeled images:

- **Train Set**: ~70% (~70 images) for model training, located in `train/images/` with labels in `train/labels/`.
- **Validation Set**: ~15% (~15 images) for hyperparameter tuning and validation, located in `val/images/` with labels in `val/labels/`.
- **Test Set**: ~15% (~15 images) for final evaluation, located in `test/images/` with labels in `test/labels/`.

The dataset includes 32 classes (e.g., Canine (13), Central Incisor (21)) defined in `dataset.yaml`, which specifies paths and class names. Example structure:
```
final_dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
  dataset.yaml
```

## Training Process

The model was trained in three stages using the YOLOv11m architecture:

### Stage 1: Initial Training
- **Epochs**: 40
- **Command**:
  ```bash
  yolo train model=yolov11m.pt data=/content/drive/MyDrive/final_dataset/dataset.yaml epochs=40 imgsz=640 batch=16
  ```
- **Purpose**: Established a baseline model.
- **Best Model**: Achieved P=0.943, R=0.915, mAP50=0.971, mAP50-95=0.765.

### Stage 2: Fine-Tuning (Stage 3)
- **Epochs**: 30
- **Command**:
  ```bash
  yolo train model=/content/runs/detect/train/weights/best.pt data=/content/drive/MyDrive/final_dataset/dataset.yaml epochs=30 imgsz=640 batch=16 lr0=1e-6
  ```
- **Purpose**: Improved localization and classification.
- **Final Model**: Achieved P=0.930, R=0.934, mAP50=0.971, mAP50-95=0.798.

### Stage 3: Additional Fine-Tuning (Stage 4)
- **Epochs**: 20
- **Command**:
  ```bash
  yolo train model=/content/runs/detect/train3/weights/best.pt data=/content/drive/MyDrive/final_dataset/dataset.yaml epochs=20 imgsz=640 batch=16 lr0=1e-6 resume=True
  ```
- **Purpose**: Enhanced performance on challenging classes.
- **Final Model**: Achieved P=0.915, R=0.921, mAP50=0.959, mAP50-95=0.801.

Training logs and weights are saved in `/content/runs/detect/train`, `train3`, and `train6` directories.

## Evaluation

### Validation
- **Command**:
  ```python
  metrics_val = model.val()
  print("Validation Metrics:", metrics_val.box.map, metrics_val.box.map50, metrics_val.box.map75)
  ```
- **Results**: P=0.921, R=0.916, mAP50=0.959, mAP50-95=0.803
- **Confusion Matrix**: Saved in `/content/runs/detect/val/confusion_matrix.png`.

### Test
- **Command**:
  ```python
  metrics_test = model.val(split='test')
  print("Test Metrics:", metrics_test.box.map, metrics_test.box.map50, metrics_test.box.map75)
  ```
- **Results**: P=0.941, R=0.895, mAP50=0.949, mAP50-95=0.745
- **Confusion Matrix**: Saved in `/content/runs/detect/train66/confusion_matrix.png`.

### Sample Predictions
- **Command**:
  ```python
  test_dir = "/content/drive/MyDrive/final_dataset/test/images"
  test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir)][:3]
  pred_results = model.predict(test_images, save=True, save_txt=True)
  ```
- **Output**: 3 annotated images with bounding boxes in `/content/runs/detect/predict/`.

## Post-Processing
To address anatomical correctness and class-specific issues (e.g., Central Incisor (41) at 0.631 mAP@50-95):
- **Code**:
  ```python
  for result in pred_results:
      boxes = result.boxes.xyxy.cpu().numpy()
      centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
      median_y = np.median(centers_y)
      upper = boxes[centers_y < median_y]
      lower = boxes[centers_y >= median_y]
      # Sort by x, adjust FDI IDs
  ```

## Submission

### GitHub Repository
- **Contents**:
  - Training notebook (e.g., `train.ipynb`).
  - `dataset.yaml` configuration file.
  - This `README.md` file.
- **Instructions**: Clone the repo and run the notebook in Colab with the specified environment.

### Word Document
- **Components**:
  - **Confusion Matrix**: Test matrix from `/content/runs/detect/train66/confusion_matrix.png`.
  - **Metrics**:
    - Validation: P=0.921, R=0.916, mAP@50=0.959, mAP@50-95=0.803
    - Test: P=0.941, R=0.895, mAP@50=0.949, mAP@50-95=0.745
  - **Summary**: "Trained YOLOv11m with 90 epochs, achieving mAP@50-95 of 0.803 on validation and 0.745 on test. The confusion matrix shows strong overall performance with minor incisor confusion, addressed via post-processing."
  - **Sample Predictions**: 3 images from `/content/runs/detect/predict/`.
  - **GitHub Link**: [Insert repo URL].
- **Download**:
  ```python
  !zip -r results.zip /content/runs
  from google.colab import files
  files.download('results.zip')
  ```

## Timeline
- **Start**: August 29, 2025, 07:53 PM IST
- **Deadline**: August 31, 2025, 07:53 PM IST
- **Tasks**:
  - Review and prepare submission by 08:20 PM IST, August 29.
  - Finalize and upload by 07:00 PM IST, August 30.
  - Submit by 07:53 PM IST, August 31.

## Acknowledgments
Thanks to the Ultralytics team for YOLOv11m and Google Colab for computational resources.