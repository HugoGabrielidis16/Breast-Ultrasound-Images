import tensorflow as tf
from model.UNET import UNET
from processingdata import load


if __name__ == "__main__":
    train_ds, test_ds = load()
    model = UNET((256, 256, 1))
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=5,
        steps_per_epoch=len(train_ds),
    )
    model.save("UNET.H5")
