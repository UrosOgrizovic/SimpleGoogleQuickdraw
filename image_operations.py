import numpy as np
from PIL import Image
from io import BytesIO
import base64
from matplotlib import pyplot as plt


def load_images(file_path, how_many=-1, is_transfer_learning=False):
    """

    :param file_path:
    :param how_many: number of images to load from file (loads all images for -1)
    :param is_transfer_learning: if true, images are padded before returning
    :return: array of loaded 28x28 images
    """

    img_array = np.load(file_path)
    if how_many != -1:
        img_array = img_array[:how_many]
    if is_transfer_learning:
        return [pad_image(image) for image in img_array]
    return [np.reshape(image, (28, 28)) for image in img_array]


def pad_image(img):
    """
    no transfer learning models that are available out-of-the-box in keras accept images with
    a resolution smaller than 32x32, hence the padding
    :param img:
    :return:
    """
    img = np.reshape(img, (28, 28))
    img = np.pad(img, 2)
    return img


def get_image_from_images(images, index):
    return images[index]


def display_image(img):
    fig = plt.figure()
    # plt.clf() clears the entire current figure
    plt.clf()

    plt.plot()
    plt.imshow(img, cmap='gray')
    plt.show()


def save_as_image(path, img_array):
    np.save(path, img_array)


def decode_base64(base64_img):
    base_64_img = base64_img.split(',')[1]
    return base64.b64decode(base_64_img)


def convert_base64_to_numpy_array(decoded_base64):
    return np.array(Image.open(BytesIO(decoded_base64)))


if __name__ == "__main__":
    load_images('data/full_numpy_bitmap_airplane.npy')