# SimpleGoogleQuickdraw
Soft Computing project, Software Engineering and Information Technologies, FTN, 2019

Technologies used: Keras 2.3.1, Python 3.6.1, Tensorflow 2.0.0

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

13 layers, excluding the input layer. Trained on both 10,000 images per label and 100,000 images per label. The latter case brought no noticeable improvement. [View architecture visualization](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_model%20architecture.svg)

[Callbacks](https://keras.io/callbacks/) used:

- [ImageDataGenerator](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L238) was used for augmenting the images, which helps avoid overfitting.

- [EarlyStopping](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L733) was especially useful for the 100k-images-per-label-model, as it greatly reduced the number of epochs that the model would execute before stopping. It was set up in such a way that if the validation loss was noticed to have stopped decreasing after five epochs, the training would terminate.

- [ModelCheckpoint](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L633) was used with the `save_best_only` flag set to `True`, so as to only save the latest best model (i.e. the best model out of all the epochs) according to the validation loss.

- [ReduceLROnPlateau](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L946) was used because models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. Yet again, the monitored value was the validation loss.

[Constraints](https://keras.io/constraints/) used:

- [MaxNorm](https://github.com/keras-team/keras/blob/master/keras/constraints.py#L22) is a type of [weight constraint](https://arxiv.org/pdf/1602.07868.pdf)<sup>1</sup>. From [Dropout: A Simple Way to Prevent Neural Networks from
Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf): *"One particular form of regularization was found to be especially useful for dropout—constraining the norm of the incoming weight vector at each hidden unit to be upper bounded by a fixed constant c."* Weight normalization works by decoupling the norm of the weight vector from the direction of the weight vector. This decoupling accelerates the convergence of stochastic gradient descent.

<sup>1</sup>For CNNs, weight normalization is computationally cheaper than [batch normalization](https://arxiv.org/pdf/1502.03167.pdf), because the number of inputs tends to be larger than the number of weights. For example, for the Vanilla CNN from this project, the number of weights is ~1.5 million, whereas there are 10k 28x28 images, which amounts to an input size of ~8 million, or, in the case of there being 100k 28x28 images, an input size of ~80 million. Furthermore, convolutions use the same filter in multiple locations (in the sense that the filter slides over the input data for that layer), which means that going through all the weights is much faster than going through all the images.

Plots:

- [10k per label train/val accuracy](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_10k_train_val_acc.png)

- [10k per label train/val loss](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_10k_train_val_loss.png)

- [100k per label train/val accuracy](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_100k_train_val_acc.png)

- [100k per label train/val loss](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/vanilla_cnn/vanilla_cnn_100k_train_val_loss.png)

## SVM 

Training was very slow; [from docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html): *"The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples."* Doesn't work well on this problem.

## [VGG19](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)

24 layers, excluding the input layer. However, instead of using VGG19's fully connected layers, I used my own, because my problem doesn't have 1000 classes. Additionally, I had to pad Google's 28x28 images to 32x32 images, because this model doesn't accept images smaller than 32x32. [View architecture visualization](https://github.com/UrosOgrizovic/SimpleGoogleQuickdraw/blob/master/models/transfer_learning/VGG19%20architecture.svg)
