import pandas as pd
import numpy as np
import image_operations
import cv2
file_path_prefix = '../data/full_numpy_bitmap_'
labels = {'airplane': np.uint8(0), 'alarm clock': np.uint8(1), 'axe': np.uint8(2), 'The Mona Lisa': np.uint8(3)}



def load_data(number_of_images_to_load_per_label):
    X = []
    Y = []
    for lab in labels:
        for img in image_operations.load_images(file_path_prefix + lab + '.npy', number_of_images_to_load_per_label):
            X.append(img)
            Y.append(labels[lab])

    return X, Y

if __name__ == "__main__":
    pass