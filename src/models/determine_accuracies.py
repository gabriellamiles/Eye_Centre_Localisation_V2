" Gabriella Miles, Farscope Phd, Bristol Robotics Laboratory"

# aim of script is to:
# 1. load pretrained models - done
# 2. load validation dataset corresponding to model - done
# 3, get model to make predictions on validation set - done
# 4. visualise these predictions - done
# 5. compare the performance of the predictions relative to the ground truth - done
# 6. inspect predictions that are erroneous by a distance of greater than 4 pixels and... - done
# 7. save a list of names of incorrectly labelled data to be corrected - done

import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import model_utils, config
from eye_data import Dataset

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError

test_num = "41"

saved_model_filepaths = os.path.join(os.getcwd(), "models")
all_saved_model_directories= os.listdir(saved_model_filepaths)
test_configuration = os.path.join(os.getcwd(), "src", "models", "model_training_runs.csv")
img_folder = os.path.join(os.getcwd(), "data", "processed", "mnt", "eme2_square_imgs")

# load training runs
config_df = pd.read_csv(test_configuration)

# load labels
labels = model_utils.load_labels() # make sure label_folder is defined as correct location (in config file)
labels = labels.sample(frac=1, random_state=42) # shuffle labels

# initialise dataset
dataset = Dataset(labels=labels)
dataset.remove_blinks()
dataset.get_train_val_test(dataset.eye_centres_no_blinks) # split into train/test data
dataset.get_k_folds() # apply k fold cross validation to training sets (folds = 5)

# for each pre-trained model
for row in range(config_df.shape[0]):

    configuration = config_df.iloc[row, :]
    model_directory = "test_" + str(configuration["Test Number"]) + "_" + configuration["Model"]

    if test_num not in model_directory:
        continue

    print("222222222222222222222222")
    print(model_directory)

    full_model_directory = ""

    for directory in all_saved_model_directories:

        # get name of directory corresponding to configuration given
        if model_directory in directory:
            full_model_directory = os.path.join(saved_model_filepaths, directory)
            break

    # retrieve most recent saved model from that file
    saved_models = sorted([i for i in os.listdir(full_model_directory) if ".hdf5" in i])
    most_recent_model = ''

    try:
        most_recent_model = os.path.join(full_model_directory, saved_models[-1])
    except IndexError:
        print("Index Error")
        continue

    print(most_recent_model)

    test_model = tf.keras.models.load_model(most_recent_model, compile=False)
    test_model.compile(loss = "mse", optimizer=Adam(learning_rate=1e-4))


    # load corresponding validation set
    # get relevant training and validation folds
    train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[configuration["Fold"]]], dataset.train_labels.iloc[dataset.kf_val_indices[configuration["Fold"]]]
    val_filenames, val_labels = val_labels[["filename"]], val_labels[config.eye_centre_cols]

    # load images
    val_images = []
    for filepath in range(val_filenames.shape[0]):

        img_filepath = os.path.join(img_folder, val_filenames.iloc[filepath, 0])
        image = tf.keras.preprocessing.image.load_img(img_filepath, color_mode="rgb", target_size=(configuration["Input Shape"], configuration["Input Shape"]))
        image = tf.keras.preprocessing.image.img_to_array(image).astype(int)
        val_images.append(image)

    val_images = np.array(val_images)

    # make predictions on images defined by val_filenames
    y_pred = test_model.predict(
        val_images,
        batch_size=1,
        verbose=1
    )

    # print(y_pred)

    # visualise these predictions (with ground truth = green, predictions = red)
    nrows, ncols = 4, 2

    fig = plt.figure(figsize=(10,10))
    for i in range(8):
        ax = fig.add_subplot(nrows, ncols, i+1)
        img_filepath = img_filepath = os.path.join(img_folder, val_filenames.iloc[i, 0])
        image = tf.keras.preprocessing.image.load_img(img_filepath, color_mode="rgb", target_size=(960, 960))
        image = tf.keras.preprocessing.image.img_to_array(image).astype(int)
        plt.imshow(image.astype('uint8'))

        # ground truth circles
        circ1 = Circle((val_labels.iloc[i, 0], val_labels.iloc[i, 1]),10, color='b')
        circ2 = Circle((val_labels.iloc[i, 2], val_labels.iloc[i, 3]),10, color='b')

        # predicted circles
        circ3 = Circle((y_pred[i, 0]*960, y_pred[i, 1]*960),10, color='g')
        circ4 = Circle((y_pred[i, 2]*960, y_pred[i, 3]*960),10, color='g')

        ax.add_patch(circ1)
        ax.add_patch(circ2)
        ax.add_patch(circ3)
        ax.add_patch(circ4)

        plt.axis(False)

    plt.savefig(os.path.join(full_model_directory, "predictions.png"))

    # determine accuracy of predictions
    val_labels_test = np.array(val_labels)
    y_pred = y_pred*960
    # print(y_pred.shape, val_labels_test.shape)

    # calculate accuracy
    diff = val_labels_test-y_pred
    squared = np.square(diff)

    left_eye = np.sum(squared[:, 0:2], axis=1)
    right_eye = np.sum(squared[:,2:4], axis=1)

    left_eye_accuracy = np.sqrt(left_eye)
    left_eye_acc = np.sum(left_eye_accuracy)/y_pred.shape[0]

    right_eye_accuracy = np.sqrt(right_eye)
    right_eye_acc = np.sum(right_eye_accuracy)/y_pred.shape[0]

    d = {'left eye accuracy': [left_eye_acc], 'right eye accuracy': [right_eye_acc]}
    data = pd.DataFrame(d)
    data.to_csv(os.path.join(full_model_directory, "results.csv"))

    print("***************ACCURACY:")
    print(full_model_directory)
    print(left_eye_acc, right_eye_acc)

    # get filenames of all predictions of error greater than 4 pixels
    print(left_eye_accuracy.shape, right_eye_accuracy.shape)

    # output ground truth relative to predictions
    columns = ["filename", "lx", "ly", "rx", "ry", "pred_lx", "pred_ly", "pred_rx", "pred_ry"]
    vals = np.concatenate((val_labels_test, y_pred), axis=1)
    data = pd.DataFrame(vals)
    val_labels_test = pd.DataFrame(val_labels_test)

    val_filenames.reset_index(inplace=True, drop=True)
    data.reset_index(inplace=True, drop=True)

    print(type(val_labels_test), type(data), type(val_filenames))
    print(val_labels_test.shape, data.shape, val_filenames.shape)

    data_df = pd.concat([val_filenames, data], axis=1)
    data_df.columns=columns
    data_df.to_csv(os.path.join(full_model_directory, "errors.csv")) # save filenames to inspect to csv file

    # produce dataframe of "most wrong" error greater than 5 pixels
    data_df["diff_lx"] = (data_df["lx"]-data_df["pred_lx"]).abs()
    data_df["diff_ly"] = (data_df["ly"]-data_df["pred_ly"]).abs()
    data_df["diff_rx"] = (data_df["rx"]-data_df["pred_rx"]).abs()
    data_df["diff_ry"] = (data_df["ry"]-data_df["pred_ry"]).abs()

    # get biggest errors
    data_large_error = data_df[(data_df["diff_lx"] > 10) | (data_df["diff_rx"] > 10) | (data_df["diff_ly"] > 10) | (data_df["diff_ry"] > 10)]
    print(data_large_error.shape, data_df.shape)
    data_large_error.to_csv(os.path.join(full_model_directory, "biggest_errors.csv"))
