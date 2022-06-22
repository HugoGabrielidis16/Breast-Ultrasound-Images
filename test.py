import tensorflow as tf

MODEL_PATH = "/model"


if __name__ == "__main__":
    UNET = tf.kera.models.load_model(f"{MODEL_PATH}/UNET.h5")
