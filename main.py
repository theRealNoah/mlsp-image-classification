# Machine Learning for Signal Processing
# Course Project - Convolutional Neural Network
# Noah Hamilton and Samuel Washburn

# Import packages we'll need for the CNN.
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
import shutil
from matplotlib import pyplot as plt
import pydot
import pydotplus
import graphviz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils.vis_utils import plot_model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

# The CNN will be written entirely in main.
if __name__ == '__main__':

    # Directory with the training data.
    training_data_dir = r'C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\Mach Learning\ML PROJECT\data_folders\train'

    # Directory with the test data.
    test_data_dir = r'C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\Mach Learning\ML PROJECT\data_folders\test'

    # Directory to save log files from the model.
    model_logs_dir = r'C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\Mach Learning\ML PROJECT\logs'

    # Directory of models that have already been trained.
    existing_models_dir = r'C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\Mach Learning\ML PROJECT\existing_models'

    # Load in the training data.
    # Instantiate an iterator for the training data.
    training_data = tf.keras.utils.image_dataset_from_directory(training_data_dir, color_mode='grayscale')
    training_data_iterator = training_data.as_numpy_iterator()
    training_batch = training_data_iterator.next()

    # Load in the test data.
    # Instantiate an iterator for the test data.
    test_data = tf.keras.utils.image_dataset_from_directory(test_data_dir, color_mode='grayscale')
    test_data_iterator = test_data.as_numpy_iterator()
    test_batch = test_data_iterator.next()

    # Normalize the training data.
    training_data = training_data.map(lambda x, y: (x / 255, y))
    training_data.as_numpy_iterator().next()

    # Normalize the test data.
    test_data = test_data.map(lambda x, y: (x / 255, y))
    test_data.as_numpy_iterator().next()

    # We will take 20% of the training data to be the validation set.
    training_size = int(len(training_data) * .8)
    validation_size = int(len(training_data) * .2)

    # Separate the training data into the training images and the validation images.
    training_images = training_data.take(training_size)
    validation_images = training_data.skip(training_size).take(validation_size)

    # Container to hold the layers of the CNN model.
    model = Sequential()

    # Add a convolutional layer to the CNN.
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 1)))
    # Add a pooling layer to the CNN.
    model.add(MaxPooling2D())
    # Add a convolutional layer to the CNN.
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    # Add a pooling layer to the CNN.
    model.add(MaxPooling2D())
    # Add a convolutional layer to the CNN.
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    # Add a pooling layer to the CNN.
    model.add(MaxPooling2D())
    # Add a flattening layer to the CNN.
    model.add(Flatten())
    # Add a dense layer to the CNN.
    model.add(Dense(256, activation='relu'))
    # Add a dense layer to the CNN.
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model using the binary crossentropy loss function.
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    # Print a summary of the CNN model.
    model.summary()

    # Callbacks is a feature of the tensorflow library for logging.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=model_logs_dir)

    # Train the CNN model that was constructed above.
    model_outputs = model.fit(training_images, epochs=20, validation_data=validation_images, callbacks=[tensorboard_callback])

    # Plot the training loss and validation loss during training.
    fig = plt.figure()
    plt.plot(model_outputs.history['loss'], color='teal', label='loss')
    plt.plot(model_outputs.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    # Plot the training accuracy and the validation accuracy during training.
    fig = plt.figure()
    plt.plot(model_outputs.history['accuracy'], color='teal', label='accuracy')
    plt.plot(model_outputs.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    # # Load a previous model that was already trained.
    # # existing_model = load_model(os.path.join(existing_models_dir, 'pneu_classifier.h5'))

    # Predict the identity of the test data using the trained model.
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    for batch in test_data.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    # Print the precision, recall, and accuracy results from predicting the test data.
    print(pre.result(), re.result(), acc.result())

    # Save the model that was just trained.
    model.save(os.path.join(existing_models_dir, 'FinalRun1.h5'))

    # # Generate a plot of the model.
    # model_img_file = r'C:\Users\Samuel Washburn\Documents\JHU Masters\Spring 2023\Mach Learning\ML PROJECT\existing_models\model_pic.png'
    # plot_model(model, to_file=model_img_file, show_shapes=True)


