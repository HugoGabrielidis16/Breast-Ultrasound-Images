import cv2
import tensorflow as tf
from glob import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split

PATH = "Dataset_BUSI_with_GT"
BENIGN_PATH = f"{PATH}/benign"


def collect_data(name):
    """
    For a classname create the two path list of images and masks
    """
    all_images = sorted(
        glob(os.path.join(PATH, name + "/*).png")),
        key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
    )
    all_masks = sorted(
        glob(os.path.join(PATH, name + "/*mask*")),
        key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
    )
    # print(all_masks[:4])
    return all_images, all_masks


def group_maks(masks):
    """
    Read the mask file and combine the masks images when their are several masks for a
    single image ( after
    """
    new_masks = []
    retard = 0
    for i in range(len(masks)):
        m = re.search(r"mask_[0-9]", masks[i])
        if m:
            new_masks[i - (retard + 1)] += cv2.resize(
                cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE), (256, 256)
            )
            retard += 1
        else:
            new_masks.append(
                cv2.resize(
                    cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE), (256, 256)
                )  # resize needed because the image need to have the same fuckign size bro
            )

    return new_masks


def load_dataset():
    """
    Combine the previous function and differents class label and apply train_test_split on it

    Retuns :
        - X_train() : the matrix of training images
        - y_train() : the matrix of training masks
        - X_test() : the matrix of testing images
        - y_test() : the matrix of testing masks
    """
    all_benign_images, all_benign_masks = collect_data("benign")

    # Here all = tf.constant(all_benign_masks) works
    all_benign_masks = group_maks(all_benign_masks)
    # Here it doesnt
    all_malignant_images, all_malignant_masks = collect_data("malignant")
    all_malignant_masks = group_maks(all_malignant_masks)

    all_images = all_benign_images + all_malignant_images
    all_masks = all_benign_masks + all_malignant_masks

    X_train, X_test = train_test_split(
        all_images,
        test_size=0.2,
        random_state=1,
        shuffle=True,
    )
    y_train, y_test = train_test_split(
        all_masks,
        test_size=0.2,
        random_state=1,
        shuffle=True,
    )
    return (X_train, y_train), (X_test, y_test)


def show_some_images(images, masks):
    """
    Show a numbers of images and it's associated masks
    """

    random_number = randint(0, len(images) - 6)
    # random_number = 192 # Used to check on malignant
    plt.figure(figsize=(20, 10))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[random_number + i + 1], "gray")
        plt.axis("off")
        plt.title("Real Image")
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        plt.imshow(masks[random_number + i + 1], "gray")
        plt.title("Mask Image")
        plt.axis("off")
    plt.show()


def process_image(image, HEIGHT=256, WIDTH=256):
    image = image.decode("utf-8")
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (HEIGHT, WIDTH))
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=-1)
    return image


def process_mask(mask):
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def preprocess(x, y):
    def f(x, y):
        image = process_image(x)
        mask = process_mask(y)
        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([256, 256, 1])
    mask.set_shape([256, 256, 1])
    return image, mask


def tf_dataset(x, y, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess)
    ds = ds.batch(batch_size)
    return ds


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_dataset()
    train_ds = tf_dataset(X_train, y_train)
    test_ds = tf_dataset(X_test, y_test)
    for x, y in train_ds.take(1):
        show_some_images(x, y)
