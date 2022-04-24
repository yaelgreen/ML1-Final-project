from typing import Dict, List

import requests
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint


def subset_of_data_set(data_set, indices):
    return {
        b'batch_label': data_set[b'batch_label'],
        b'labels': [data_set[b'labels'][i] for i in indices],
        b'data': np.take(data_set[b'data'], indices, axis=0),
        b'filenames': [data_set[b'filenames'] for i in indices],
    }


def random_indices(choose: int, out_of: int) -> List[int]:
    return [randint(0, out_of) for i in range(0, choose)]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def mapping(x):
    return (-1 + (2 / 255) * x)


def download_and_extract_cifar_10_dataset(
        url: str = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz") -> None:
    filename = "cifar-10-python.tar.gz"

    # Download the CIFAR 10 dataset
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.raw.read())

    # Extract the tar.gz file
    import shutil
    extract_path = "cifar-10"
    shutil.unpack_archive(filename, extract_path)


def create_training_and_validation_sets(training_size: str = 1000) -> \
        (Dict, Dict, Dict):
    # load dataset
    path = os.path.join(os.getcwd(), "cifar-10", "cifar-10-batches-py")
    first_batch = unpickle(os.path.join(path, "data_batch_1"))
    # choose random 1000 images
    indices = random_indices(9999, training_size)
    training_set = subset_of_data_set(first_batch, indices)
    validation_set = unpickle(os.path.join(path, "data_batch_2"))
    meta = unpickle(os.path.join(path, "batches.meta"))
    return training_set, validation_set, meta


def plot_image(data, meta, image_index=0):
    # get image and RGB channels from dataset
    image = data[b'data'][image_index, :]
    image_r = image[0:1024].reshape(32, 32)
    image_g = image[1024:2048].reshape(32, 32)
    image_b = image[2048:].reshape(32, 32)
    # plot image using RGB channels
    img = np.dstack((image_r, image_g, image_b))
    title = meta[b'label_names'][data[b'labels'][image_index]]
    return img, title


def plot_random_images(data_set: Dict, meta: Dict, plt_title="plot",
                       number_of_images: int = 50) -> None:
    # choose random 50 images
    indices = random_indices(999, number_of_images)
    # plot images
    n_col = 10
    n_row = int(number_of_images / n_col)
    imgs = [plot_image(data_set, meta, image_index) for image_index in indices]
    _, axs = plt.subplots(n_row, n_col, figsize=(32, 32))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img[0])
        ax.set_title(img[1])
    plt.subplots_adjust(top=0.9)
    plt.suptitle(plt_title)
    plt.show()
    plt.savefig(f"{plt_title}.png")


def convert_pixel_intensity(data_set):
    data_set[b'data'] = mapping(data_set[b'data'])


if __name__ == "__main__":
    download_and_extract_cifar_10_dataset()
    training_set, validation_set, meta = create_training_and_validation_sets()
    plot_random_images(training_set, meta, "training_set_images")
    plot_random_images(validation_set, meta, "validation_set_images")
    convert_pixel_intensity(training_set)
    convert_pixel_intensity(validation_set)



