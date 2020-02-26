# SimpleGoogleQuickdraw
Soft Computing project, Software Engineering and Information Technologies, FTN, 2019

Technologies used: Keras 2.3.1, Python 3.6.1, Tensorflow 2.0.0

# Demo

<a href="https://media.giphy.com/media/Ri2xsHaPlKv2sbv4pz/giphy.gif"><img src="https://media.giphy.com/media/Ri2xsHaPlKv2sbv4pz/giphy.gif"/></a>

# How to run

- clone the project via `git clone https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw.git`

- download data (see [Fetching the data](#fetching-the-data))

- create virtual environment (`virtualenv venv`)

- in terminal, enter `set WRAPT_INSTALL_EXTENSIONS=false` (this is required due to a `pip install tensorflow` problem)

- in terminal, enter `pip3 install -r requirements.txt` to install the dependencies

- download VGG weights: https://drive.google.com/file/d/1uvpi0ugDtwueWnGk4m13jp3KK8HAbVCj/view?usp=sharing and place them in `models/transfer_learning`

- run `web.py`

# Fetching the data

Create a folder called `data` in project root, download and place the following files into that folder:

Airplane: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/airplane.npy

Alarm clock: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/alarm%20clock.npy

Ant: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/ant.npy

Axe: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/axe.npy

Bicycle: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/bicycle.npy

The Mona Lisa: https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/The%20Mona%20Lisa.npy

# Models

## Vanilla CNN

13 layers, excluding the input layer ([view architecture visualization](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_model%20architecture.svg)). Dropout was used to avoid overfitting. The kernel's dimensions are 3x3, which is an often-used kernel size. 

This model was trained on both 10,000 images per label and 100,000 images per label. The latter case brought no noticeable improvement. 

[Callbacks](https://keras.io/callbacks/) used:

- [ImageDataGenerator](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L238) was used for augmenting the images, which helps avoid overfitting.

- [EarlyStopping](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L733) was especially useful for the 100k-images-per-label-model, as it greatly reduced the number of epochs that the model would execute before stopping. It was set up in such a way that if the validation loss was noticed to have stopped decreasing after five epochs, the training would terminate.

- [ModelCheckpoint](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L633) was used with the `save_best_only` flag set to `True`, so as to only save the latest best model (i.e. the best model out of all the epochs) according to the validation loss.

- [ReduceLROnPlateau](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L946) was used because models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates [2]. Yet again, the monitored value was the validation loss.

[Constraints](https://keras.io/constraints/) used:

- [MaxNorm](https://github.com/keras-team/keras/blob/master/keras/constraints.py#L22) is a type of [weight constraint](https://arxiv.org/pdf/1602.07868.pdf).<sup>1</sup> From [Dropout: A Simple Way to Prevent Neural Networks from
Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf): *"One particular form of regularization was found to be especially useful for dropout—constraining the norm of the incoming weight vector at each hidden unit to be upper bounded by a fixed constant c."* 

Plots:

- [10k per label train/val accuracy](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_10k_train_val_acc.png)

- [10k per label train/val loss](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_10k_train_val_loss.png)

- [100k per label train/val accuracy](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_100k_train_val_acc.png)

- [100k per label train/val loss](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_100k_train_val_loss.png)

## SVM 

The default value of 1 was used for *C*, the penalty error term. 'rbf' was the value used for the *kernel* parameter, also by default. The default value of 'scale' was used for *gamma*, the RBF kernel coefficient.<sup>2</sup>

Training was very slow; [from docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html): *"The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples."* This model doesn't work well on this problem.

Perhaps the performance of this model could be improved by performing a "grid search" on *C* and *gamma* using cross-validation [1].

<sup>2</sup>*C* tells the SVM optimization how much to avoid misclassifying each training example by (large *C* - small hyperplane, and vice versa), and *gamma* defines how far the influence of a single training example (i.e. point) reaches (large *gamma* - the decision boundary will only depend on the points close to it - that is, each point's influence radius will be small, and vice versa).

## [VGG19](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)

Consists of 24 layers, excluding the input layer ([view architecture visualization](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/transfer_learning/VGG19%20architecture.svg)). However, instead of using VGG19's fully connected layers, I used my own, because my problem doesn't have 1000 classes. Additionally, I had to pad Google's 28x28 images to 32x32 images, because this model doesn't accept images smaller than 32x32. 

This model uses 3x3 convolution filters. Its predecessor, VGG16, achieved state-of-the-art results in the ImageNet Challenge 2014 by adding more weight layers compared to previous models that had done well in that competition.

## Accuracy-per-model

set | CNN 10k  | CNN 100k | SVM 2k | SVM 10k | VGG 10k | VGG 100k |
--- | -------- | -------- | ------ | ------- | ------- | -------- |
train | ~60%  | ~85%  | ~85%  | ~82%  | ~90% | ~93% |
validaiton | ~70%  | ~85%  | N/A*  | ~84%  | ~95% | ~94% |

\* For SVM 2k, the data was not divided into train and validation sets, as there were too few examples

# References
[1] - [Hsu, Chih-Wei, Chih-Chung Chang, and Chih-Jen Lin. "A practical guide to support vector classification." (2003): 1396-1400.](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)

[2] - [Ravaut, Mathieu, and Satya Gorti. "Gradient descent revisited via an adaptive online learning rate." arXiv preprint arXiv:1801.09136 (2018).](https://arxiv.org/pdf/1801.09136.pdf)
