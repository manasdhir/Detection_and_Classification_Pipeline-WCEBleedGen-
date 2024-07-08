import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_data(data_dir):
    X = []
    Y = []

    for category in os.listdir(data_dir):
        path = os.path.join(data_dir, category, 'Images')
        for images in os.listdir(path):
            imagePath = os.path.join(path, images)
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            X.append(np.array(image))
            if category == 'bleeding':
                Y.append(1.0)
            else:
                Y.append(0.0)

    return np.array(X) / 255.0, to_categorical(LabelEncoder().fit_transform(Y), 2)

def split_data(X, Y):
    x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.20, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42)
    return x_train, x_val, x_test, y_train, y_val, y_test

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Loader")
    parser.add_argument("data_dir", type=str, help="Directory of the dataset")
    args = parser.parse_args()

    X, Y = get_data(args.data_dir)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(X, Y)

    print("Data Loaded Successfully")
    print(f"Training Data Shape: {x_train.shape}, {y_train.shape}")
    print(f"Validation Data Shape: {x_val.shape}, {y_val.shape}")
    print(f"Test Data Shape: {x_test.shape}, {y_test.shape}")
