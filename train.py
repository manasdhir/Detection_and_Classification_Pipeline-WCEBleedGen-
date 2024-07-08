from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_loader import get_data, split_data
from model import build_classification_model
import argparse

parser = argparse.ArgumentParser(description="Train the CNN model")
parser.add_argument("data_directory", type=str, help="Path to the data directory")
parser.add_argument("weights_path", type=str, help="Path to save the model weights")
args = parser.parse_args()

X, Y = get_data(args.data_directory)
x_train, x_val, x_test, y_train, y_val, y_test = split_data(X, Y)

model = build_classification_model()

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

model_checkpoint_callback = ModelCheckpoint(
    filepath=args.weights_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='auto',
    verbose=1,
    save_best_only=True
)

batch_size = 64
epochs = 20

model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          epochs=epochs, validation_data=(x_val, y_val),
          verbose=1, steps_per_epoch=x_train.shape[0] // batch_size,
          callbacks=[model_checkpoint_callback])
