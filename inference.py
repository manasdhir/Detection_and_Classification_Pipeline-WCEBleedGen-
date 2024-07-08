import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import build_classification_model
import torch
import argparse

parser = argparse.ArgumentParser(description="Inference Script")
parser.add_argument("images_dir", type=str, help="Directory containing input images")
parser.add_argument("classification_model_path", type=str, help="Path to the classification model weights")
parser.add_argument("detection_weights_path", type=str, help="Path to the YOLOv8 weights")
args = parser.parse_args()

def get_data(data_dir):
    X = []
    names = []

    for image in sorted(os.listdir(data_dir)):
        names.append(image)
        imagePath = os.path.join(data_dir, image)
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        X.append(np.array(img))

    return np.array(X) / 255.0, names

def classify_and_detect(images_dir, classification_model_path, detection_weights_path):
    X, names = get_data(images_dir)
    classification_model = build_classification_model()
    classification_model.load_weights(classification_model_path)
    detection_model = torch.hub.load('ultralytics/yolov8', 'custom', path=detection_weights_path)

    classifications = classification_model.predict(X)
    predictions = np.argmax(classifications, axis=1)

    results = []

    for i, prediction in enumerate(predictions):
        if prediction == 1:  # If the image is classified as bleeding
            detections = detection_model(os.path.join(images_dir, names[i]))
            image = cv2.imread(os.path.join(images_dir, names[i]))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if len(detections.xyxy) > 0:
                for detection in detections.xyxy[0]:
                    x_min, y_min, x_max, y_max, confidence, class_id = detection[:6]
                    if confidence > 0.5:
                        cv2.rectangle(image_rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                        cv2.putText(image_rgb, f"Detection {int(class_id)}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            results.append(("Bleeding", image_rgb))
        else:
            image = cv2.imread(os.path.join(images_dir, names[i]))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label_text = 'Non-Bleeding'
            color = (0, 0, 255)  # Red for non-bleeding
            cv2.putText(image_rgb, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            results.append(("Non-Bleeding", image_rgb))

    return results, names

results, names = classify_and_detect(args.images_dir, args.classification_model_path, args.detection_weights_path)

# Display the images with the detection/classification results
subplot_size = 10
fig, axs = plt.subplots(len(results) // subplot_size + (len(results) % subplot_size > 0), subplot_size, figsize=(20, 20))

for i, (label, img) in enumerate(results):
    axs[i // subplot_size, i % subplot_size].imshow(img)
    axs[i // subplot_size, i % subplot_size].axis('off')
    axs[i // subplot_size, i % subplot_size].set_title(f"{label} Image\n{names[i]}")

plt.tight_layout()
plt.show()
