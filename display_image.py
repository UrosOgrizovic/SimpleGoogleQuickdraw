import numpy as np
import matplotlib.pyplot as plt
import png


def load_images(file_path):
    img_array = np.load(file_path)
    img_array = [np.reshape(image, (28, 28)) for image in img_array]
    plt.plot()
    # plt.imshow(img_array[0], cmap='gray')
    # plt.show()

    return img_array

def get_image_from_images(images, index):
    return images[index]

def save_as_image(image):
    file = open('image.png', 'wb')
    w = png.Writer(255, 1, greyscale=True)
    w.write(file, image)

    file.close()

