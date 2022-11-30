import configparser
import os

from keras.utils import image_utils
import matplotlib.pyplot as plt
import numpy as np

config = configparser.ConfigParser()
config.read('config.ini')
IMG_SIZE = (int(config['IMG']['width']), int(config['IMG']['height']))
IMG_PATH = config['IMG']['path']


def show_image(img_path):
    """
    show a image from img_path
    """
    img = image_utils.load_img(img_path, target_size=IMG_SIZE)
    plt.imshow(img)
    plt.show()


def show_sample_images(label=False):
    files = os.listdir(IMG_PATH)
    np.random.shuffle(files)
    gender_dict = {0:'Male', 1:'Female'}

    n = 6

    fig, axes = plt.subplots(n, n, figsize=(10,10))
    for i in range(n*n):
        filename = files[i]
        img = image_utils.load_img('../data/UTKFace/' + filename)
        ax = axes.flat[i]
        ax.set_xticks([])
        ax.set_yticks([])

        if label:
            components = filename.split('_')
            age = int(components[0])
            gender = int(components[1])
            ax.set_xlabel(f'{gender_dict[gender]}: {age}')
        ax.imshow(img)

    plt.show()


if __name__ == '__main__':
    show_sample_images(True)