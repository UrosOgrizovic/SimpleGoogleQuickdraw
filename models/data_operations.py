import pandas as pd
import numpy as np
import image_operations
file_path_prefix = '../data/full_numpy_bitmap_'
labels = {'airplane': np.uint8(0), 'alarm clock': np.uint8(1), 'axe': np.uint8(2), 'The Mona Lisa': np.uint8(3)}


def load_data(number_of_images_to_load_per_label):
    """

    :param number_of_images_to_load_per_label:
    :return: dataFrame of images whose columns are 'label' and 'image'
    """
    images = {}
    rows_list = []
    for lab in labels:
        images[lab] = image_operations.load_images(file_path_prefix + lab + '.npy', number_of_images_to_load_per_label)

    for lab in images:
        for img in images[lab]:

            new_img = np.expand_dims(img, axis=-1)

            rows_list.append([labels[lab], new_img])

    return pd.DataFrame(rows_list, columns=['label', 'image'])



if __name__ == "__main__":
    data = load_data(100000)
    print(data)
