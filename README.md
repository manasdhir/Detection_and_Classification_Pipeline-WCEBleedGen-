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
   

