# SimpleGoogleQuickdraw
Soft Computing project, Software Engineering and Information Technologies, FTN, 2019

Technologies used: Keras, Python

# Fetching the data

Create a folder called `data` in project root, download and place the following files into that folder:

Airplane: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/airplane.npy

Alarm clock: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/alarm%20clock.npy

Ant: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/ant.npy

Axe: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/axe.npy

Bicycle: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/bicycle.npy

The Mona Lisa: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/The%20Mona%20Lisa.npy

# Models

**Vanilla CNN** - 13 layers, excluding the input layer. Trained on both 10.000 images per label and 100.000 images per label. The latter case brought no noticeable improvement. [View architecture visualization](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_model%20architecture.svg)

**SVM** - Training was very slow; [from docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) *"The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples."* Doesn't work well on this problem.

[**VGG19**](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py) - 24 layers, excluding the input layer. However, instead of using VGG19's fully connected layers, I used my own, because my problem doesn't have 1000 classes. [View architecture visualization](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/transfer_learning/VGG19%20architecture.svg)
