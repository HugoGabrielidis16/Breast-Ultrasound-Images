import tensorflow as tf
from processingdata import load
import matplotlib.pyplot as plt
from random import randint
import numpy as np

MODEL_PATH = "model"


if __name__ == "__main__":
    UNET = tf.keras.models.load_model(f"{MODEL_PATH}/UNETv2.h5")
    train_ds, test_ds = load()
    UNET.evaluate(test_ds)
