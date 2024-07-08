# Detection and Classification Combined Pipeline
This repository contains the scripts used to make a combined pipeline to perform Detection and Classification of bleeding region in Wireless Capsule Endoscopy Images. A research paper on the same has been presented
at IEEE ICPEICES 2024 and will be published soon.

## Repository Structure
- `data_loader.py`: Contains functions to load and preprocess image data.
- `model.py`: Defines the CNN model architecture for classification.
- `train.py`: Script to train the classification model.
- `inference.py`: Script for inference. It classifies images and runs detection if bleeding is detected.

## Usage
### Setup
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
### Training

To train the classification model, run:
```bash
python train.py --data_dir <path-to-dataset> --model_save_path <path-to-save-model>
```

Example:
```bash
python train.py --data_dir /path/to/dataset --model_save_path /path/to/save/model
```

### Inference

To run inference on a directory of images, run:
```bash
python inference.py <images_dir> <classification_model_path> <detection_weights_path>
```

Example:
```bash
python inference.py /path/to/images /path/to/classification_model.h5 /path/to/yolo_weights/best.pt
```

## Scripts Description

### `data_loader.py`

Contains the function `get_data(data_dir)` which loads and preprocesses images from the given directory.

### `model.py`

Defines the function `build_classification_model()` which constructs the CNN model used for classifying images.

### `train.py`

Script to train the classification model. It uses data from the specified directory and saves the trained model to the given path.

### `inference.py`

Script for performing inference on a batch of images. It first classifies the images and then runs YOLOv8 detection if bleeding is detected in any image. The results are displayed with appropriate labels and bounding boxes.
### Example Output of inference.py
![fig1](https://github.com/manasdhir/Detection_and_Classification_Pipeline-WCEBleedGen-/assets/142010408/f696d309-bf0d-4f31-8a15-c934498173b5)

## Notes

Ensure you have the correct paths for your datasets and model weights when running the scripts. The inference script is designed to handle multiple images and display results in a structured manner.


