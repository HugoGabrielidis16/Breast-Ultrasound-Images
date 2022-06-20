import cv2
import tensorflow as tf
from glob import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from random import randint, random
from sklearn.model_selection import train_test_split

PATH = "Dataset_BUSI_with_GT"
BENIGN_PATH = f"{PATH}/benign"


def show_some_images(images, masks):

    random_number = randint(0, len(images) - 6)
    # random_number = 192 # Used to check on malignant
    plt.figure(figsize=(20, 10))
    for i in range(5):
        plt.subplot(2, 5, +i + 1)
        plt.imshow(cv2.imread(images[random_number + i + 1]), "gray")
        plt.title("Real Image")
        plt.axis("off")
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        plt.imshow(masks[random_number + i + 1], "gray")
        plt.title("Mask Image")
        plt.axis("off")
    plt.show()


def collect_data(name):
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
            new_masks[i - (retard + 1)] += cv2.imread(masks[i])
            retard += 1
        else:
            new_masks.append(cv2.imread(masks[i]))
    return new_masks


def process_image(image, HEIGHT=256, WIDTH=256):
    image = cv2.resize(image, (HEIGHT, WIDTH))
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def process_masks(mask, HEIGHT=256, WIDTH=256):
    mask = cv2.resize(mask, (HEIGHT, WIDTH))
    return mask


def preprocess(x,y):
    

def tf_dataset(x, y, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess)


def load_dataset():
    all_benign_images, all_benign_masks = collect_data("benign")
    all_benign_masks = group_maks(all_benign_masks)
    all_malignant_images, all_malignant_masks = collect_data("malignant")
    all_malignant_masks = group_maks(all_malignant_masks)

    all_images = all_benign_images + all_malignant_images
    all_masks = all_benign_masks + all_benign_masks

    X_train, X_test = train_test_split(
        all_images,
        test_size=0.2,
        shuffle=True,
        random_state=1,
    )
    y_train, y_test = train_test_split(
        all_masks,
        test_size=0.2,
        shuffle=True,
        random_state=1,
    )

    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    ordering()
