from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import torch

def build_classification_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_detection_model(weights_path):
    detection_model = torch.hub.load('ultralytics/yolov8', 'custom', path=weights_path)
    return detection_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Loader")
    parser.add_argument("weights_path", type=str, help="Path to the YOLOv8 weights")
    args = parser.parse_args()

    detection_model = load_detection_model(args.weights_path)
    print("Detection Model Loaded Successfully")
