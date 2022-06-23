import tensorflow as tf
from processingdata import load
import matplotlib.pyplot as plt
from random import randint
import numpy as np

MODEL_PATH = "model"


def show_images(images, y_true, y_pred):
    random_number = randint(0, len(y_true) - 6)
    # random_number = 4  # Used to check on malignant
    plt.figure(figsize=(20, 10))
    for i in range(5):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[random_number + i + 1])
        plt.axis("off")
        plt.title("Original Images")

        plt.subplot(3, 5, i + 6)
        plt.imshow(y_true[random_number + i + 1], "gray")
        plt.axis("off")
        plt.title("True Mask")
    for i in range(5):
        plt.subplot(3, 5, i + 11)
        plt.imshow(y_pred[random_number + i + 1], "gray")
        plt.title("Predicted Mask")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    UNET = tf.keras.models.load_model(f"{MODEL_PATH}/UNETv2.h5")
    train_ds, test_ds = load()
    for x, y in test_ds.take(3):
        y_pred = UNET.predict(x)
        for i in range(len(y_pred)):
            y_pred[i] = np.round(y_pred[i])
        show_images(x, y, y_pred)
