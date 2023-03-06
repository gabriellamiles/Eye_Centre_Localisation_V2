"Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory"

import os
import time

import pandas as pd
import numpy as np

from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from PIL import Image

import model_utils

def load_imgs_from_list(root_directory, list):

    data = []

    for image_filepath in list:
        full_image_filepath = os.path.join(root_directory, image_filepath)
        im = Image.open(full_image_filepath)
        data.append(im)
        # im.close()

    return data

def predict_on_test_set(data, model, filenames, img_folder):

    predictions = []

    count = 0
    for image in data:

        image = np.array([image])

        # make eye centre predictions on the input image
        preds = model.predict([image], batch_size=1)
        x, y = preds[0]
        print("***************")
        print(x, y)

        # scale the eye centre coordinates based on the image dimensions
        im = Image.open(os.path.join(img_folder, filenames[count]))
        (w, h) = im.size
        x = int(x * w)
        y = int(y * h)
        print(x, y)

        predictions.append([filenames[count], x, y]) # store to save to csv file later

        count += 1

    # save predictions
    df = pd.DataFrame(predictions, columns = ['filename', 'x', 'y']) # also need bounding box coordinates

    return df

def preprocess_data(list_of_data):

    resized_data = []
    new_size = (320, 320)


    for item in list_of_data: # for every image (single eye)
        old_size = item.size # determine size of original image

        new_im = Image.new("RGB", new_size)
        box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))

        new_im.paste(item, box)

        # resized_data.append(img_to_array(new_im))

        # print(item.size)
        tmp_data = np.array(item, dtype="float32") / 255.0
        resized_data.append(tmp_data)

        item.close()
        new_im.close()

    return resized_data

def calculate_accuracy(test_df, predictions_df):

    test_targets = test_df[['x', 'y']]#.astype(np.uint8)
    predictions = predictions_df[['x', 'y']]#.astype(np.uint8)

    # add ground truth info to file
    predictions_df["x_actual"] = test_targets["x"]
    predictions_df["y_actual"] = test_targets["y"]

    # calculate error in x and y, then total error
    predictions_df["x_error"] = test_targets["x"] - predictions["x"]
    predictions_df["y_error"] = test_targets["y"] - predictions["y"]
    predictions_df["direct_error"] = (predictions_df["x_error"]**2 + predictions_df["y_error"]**2)**0.5

    # calculated average pixel error
    sum_total_error = predictions_df.direct_error.sum()
    accuracy = sum_total_error/predictions_df.shape[0]
    # calculate standard_deviation
    standard_deviation = predictions_df.direct_error.std()
    print(str(accuracy) + " +/- " + str(standard_deviation))

    # save predictions, ground truth and accuracy information to csv file
    predictionSavePath = os.path.join(os.getcwd(), "models", model_directory, "test_set_predictions.csv")
    predictions_df.to_csv(predictionSavePath)

    return accuracy, standard_deviation

def predict_on_unseen_data(img_filepaths, model):

    predictions = []

    count = 0
    for filepath in img_filepaths:
        print("Progress: " + str(count) + "/" + str(len(img_filepaths)))
        save_filename = filepath.split("/cropped_eye_imgs/")[-1]

        pil_image = load_img(filepath)

        # add black bars to image
        old_size = pil_image.size
        new_size = (324, 324)
        img_with_border = Image.new("RGB", new_size)   ## luckily, this is already black!
        box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
        img_with_border.paste(pil_image, box)
        img_with_border = img_with_border.resize((224, 224))

        image = img_to_array(img_with_border, dtype="float32") / 255.0
        image = np.expand_dims(image, axis=0)

        preds = model.predict(image, batch_size=1)
        x, y = preds[0]*224
        print("***************")
        print(x, y)

        predictions.append([save_filename, x, y])
        count +=1

    # save predictions
    df = pd.DataFrame(predictions, columns = ['filename', 'x', 'y']) # also need bounding box coordinates

    return df

if __name__ == '__main__':

    TEST_SET = 0
    UNSEEN = 1

    # initialise key filepaths
    root_folder = os.getcwd()
    img_folder = os.path.join(root_folder, "data", "processed", "test_data", "imgs")
    test_csv_filepath = os.path.join(root_folder, "data", "processed", "test_data", "labels.csv")
    unseen_folder = os.path.join(root_folder, "data", "processed", "mnt", "cropped_eye_imgs", "right_eye")

    # load and compile trained model~
    print("[INFO] Loading trained model...")
    model_directory = "20230210_173438"
    model_file = "vgg-weights-improve-19-0.000251.hdf5"
    model_filepath = os.path.join(os.getcwd(), "models", model_directory, model_file)
    model, keyword = model_utils.load_model()
    model.load_weights(model_filepath)

    # compile, and key parameters for compile function
    INIT_LR = 1e-4
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="mse", optimizer=opt)

    if TEST_SET:
        # load saved test data for model to predict on
        print("[INFO] Loading test set...")
        test_df = pd.read_csv(test_csv_filepath)[["filename", "x", "y"]]
        filenames = test_df.filename.tolist()
        data = load_imgs_from_list(img_folder, filenames)
        data = preprocess_data(data)

        # make predictions
        print("[INFO] Making predictions on test set...")
        predictions_df = predict_on_test_set(data, model, filenames, img_folder)

        # calculate accuracy on test set
        print("[INFO] Calculating performance metrics...")
        accuracy, standard_deviation = calculate_accuracy(test_df, predictions_df)

    if UNSEEN:
        participant_folders = [os.path.join(unseen_folder, directory) for directory in os.listdir(unseen_folder)]
        trial_folders = []

        for participant in participant_folders:

            if "zi" in participant:
                continue

            for trial_folder in os.listdir(participant):

                trial_folders.append(os.path.join(participant, trial_folder))

        count = 0
        # print(trial_folders)
        for trial_folder in trial_folders:

            # initialise key filepaths
            save_directory = os.path.join(root_folder, "models", model_directory, "predictions", "right_eye")
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            csv_save_name = trial_folder.split("/")[-2] + "_" + trial_folder.split("/")[-1] + ".csv"
            if os.path.exists(os.path.join(save_directory, csv_save_name)):
                print(csv_save_name + " already predicted, moving on!")
                continue
            print("[INFO] Predictions saved under: " + str(os.path.join(save_directory, csv_save_name)))

            # get list of filenames for single trial
            print("[INFO] Loading unseen data...:" + str(trial_folder))
            img_filenames = os.listdir(trial_folder)
            img_filenames = [os.path.join(trial_folder, img_filename) for img_filename in img_filenames]

            print(len(img_filenames))
            pred_df = predict_on_unseen_data(img_filenames, model)
            pred_df.to_csv(os.path.join(save_directory, csv_save_name))

            count +=1
            if count == 4:
                break
