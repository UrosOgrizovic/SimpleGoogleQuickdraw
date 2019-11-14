import numpy as np
import matplotlib.pyplot as plt
import png
from PIL import Image
from io import BytesIO
import base64

def load_images(file_path):
    img_array = np.load(file_path)
    return [np.reshape(image, (28, 28)) for image in img_array]

def get_image_from_images(images, index):
    return images[index]

def display_image(img):
    # plt.clf() clears the entire current figure
    plt.clf()

    plt.plot()
    plt.imshow(img, cmap='gray')
    plt.show()

def save_as_image(path, img_array):
    np.save(path, img_array)

def decode_base64(base64_img):
    base64Img = base64_img.split(',')[1]
    return base64.b64decode(base64Img)

def convert_base64_to_numpy_array(decoded_base64):
    return np.array(Image.open(BytesIO(decoded_base64)))

if __name__ == "__main__":
    load_images('data/full_numpy_bitmap_airplane.npy')