"Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory"

import os
import model_utils

import pandas as pd
import numpy as np

from PIL import Image

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint

def load_labels(filepath):

    labels_df = pd.read_csv(filepath)[["filename", "x", "y", "x_actual", "y_actual", "x_error", "y_error", "direct_error"]]

    # remove rows where direct error is less than 5
    print(labels_df.shape)
    labels_df = labels_df[labels_df["direct_error"] >= 5]
    print(labels_df.shape)

    return labels_df

def construct_dataset(labels_df, img_directory):

    data, filenames, targets = [], [], []

    print(labels_df.head())

    for row in range(labels_df.shape[0]):

        # extract informationn from dataframe
        filename = labels_df.iloc[row, 0]
        target_x = labels_df.iloc[row, 3]
        target_y = labels_df.iloc[row, 4]

        # load and store image data
        img_filepath = os.path.join(img_directory, filename)
        image = Image.open(img_filepath)
        w, h = image.size

        # store targets for easy conversion to dataframe
        target = [target_x/w, target_y/h]

        # add all relevant data to relevant lists
        filenames.append(filename)
        data.append(img_to_array(image))
        targets.append(target)

    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")

    return data, filenames, targets

def partition_dataset(data, targets, filenames):

    split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)

    # unpack the data split
    (train_images, test_images) = split[:2]
    (train_targets, test_targets) = split[2:4]
    (train_filenames, test_filenames) = split[4:]

    return train_images, train_targets, train_filenames, test_images, test_targets, test_filenames

if __name__ == '__main__':

    #initialise key filepaths
    model_directory = "20230204_123435"
    model_weights = "vgg-weights-improve-15-0.000449.hdf5"
    root_folder = os.getcwd() # current working directory
    labels = os.path.join(root_folder, "models", model_directory, "test_set_predictions.csv") # filepath to predictions and ground truth data
    img_directory = os.path.join(root_folder, "data", "processed", "test_data", "imgs") # filepaths to images

    # load labels with greater error than 5 pixels
    labels_df = load_labels(labels)
    # construct retraining set
    train_data, filenames, train_targets = construct_dataset(labels_df, img_directory)
    train_images, train_targets, train_filenames, test_images, test_targets, test_filenames = partition_dataset(train_data, train_targets, filenames)

    # load and compile trained model
    print("[INFO] Loading trained model...")
    model_filepath = os.path.join(root_folder, "models", model_directory, model_weights)
    model, keyword = model_utils.load_model()
    model.load_weights(model_filepath)
    # compile, and key parameters for compile function
    INIT_LR = 1e-4
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="mse", optimizer=opt)

    # Callbacks - recording the intermediate training results which can be visualised on tensorboard
    tensorboardPath = os.path.join(os.getcwd(), "models", model_directory, "retrain")
    checkpointPath = os.path.join(os.getcwd(), "models" , model_directory, "retrain")
    checkpointPath = checkpointPath + "/" + keyword + "-weights-improve-{epoch:02d}-{val_loss:02f}.hdf5"

    callbacks = [
        TensorBoard(log_dir=tensorboardPath, histogram_freq=0, write_graph=True, write_images=True),
        ModelCheckpoint(filepath=checkpointPath, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
    ]

    # retrain model on incorrect data for eye centre regression
    print("[INFO] training eye centre regressor...")
    H = model.fit(
        train_images, train_targets,
        validation_data=(test_images, test_targets),
        callbacks=callbacks,
        batch_size=8,
        epochs=10,
        verbose=1)
