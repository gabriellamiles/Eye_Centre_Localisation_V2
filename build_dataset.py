import config
import cv2
import os

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array

def build_dataset(imgFolder, centres_df, region_df, target_size):

    # initialise empty dataset
    fin_dataframe = pd.DataFrame(index=range(1),columns=['filename','lx', 'ly', 'rx', 'ry', 'startLX', 'startLY', 'endLX', 'endLY', 'startRX', 'startRY', 'endRX', 'endRY'])

    set_height = (region_df["endLX"] - region_df["startLX"]).max()

    # include parameters to adjust based on expected input shape to model
    remove = 0
    extra = 0
    diff = set_height - target_size
    remove = int(diff/2)

    if set_height > target_size:
        if diff%2 == 1:
            extra = 1
    elif set_height < target_size:
        if diff%2 == 1:
            extra = -1

    left_eye = []
    calibration = 75
    data, targets, filenames = [], [], []
    # combine centres and bounding box region for each participant
    print("[INFO] Building datasets...")
    for i in range(0, region_df.shape[0]):

        name = region_df["filename"].iloc[i]
        centres = centres_df[centres_df["filename"] == name]

        if centres.shape[0] == 0:
            # if no match is found... continue on
            continue

        left_eye_left = region_df["startLY"].iloc[i] - calibration
        left_eye_top = region_df["startLX"].iloc[i]
        left_eye_right = region_df["startLY"].iloc[i] + set_height - calibration
        left_eye_bottom = region_df["startLX"].iloc[i] + set_height

        right_eye_left = region_df["startRY"].iloc[i] - calibration
        right_eye_top = region_df["startRX"].iloc[i]
        right_eye_right = region_df["startRY"].iloc[i] + set_height - calibration
        right_eye_bottom = region_df["startRX"].iloc[i] + set_height

        new_lx = int(centres["lx"]) - left_eye_top - remove
        new_ly = int(centres["ly"]) - left_eye_left - remove
        new_rx = int(centres["rx"]) - right_eye_top - remove
        new_ry = int(centres["ry"]) - right_eye_left - remove

        if not new_lx < 0:
            im = cv2.imread(os.path.join(imgFolder, name))
            cropped_im = im[left_eye_left+remove:left_eye_right-(remove+extra), left_eye_top+remove:left_eye_bottom-(remove+extra)]
            cropped_im2 = im[right_eye_left+remove:right_eye_right-(remove+extra), right_eye_top+remove:right_eye_bottom-(remove+extra)]

            (cropped_im_h, cropped_im_w) = cropped_im.shape[:2]

            new_lx = new_lx/ cropped_im_h
            new_ly = new_ly/cropped_im_w
            new_rx = new_rx/cropped_im_h
            new_ry = new_ry/cropped_im_w

            left_image = img_to_array(cropped_im)
            right_image = img_to_array(cropped_im2)

            if cropped_im.shape[0] != target_size or cropped_im.shape[1] != target_size:
                pass
            else:
                # append left eye image
                data.append(left_image)
                targets.append((new_lx, new_ly))
                filenames.append(name)

            if cropped_im2.shape[0] != target_size or cropped_im2.shape[1] != target_size:
                pass
            else:
                # append right eye image
                data.append(right_image)
                targets.append((new_rx, new_ry))
                filenames.append(name)

    return data, targets, filenames



def partition_dataset(data, targets, filenames):

    print("Partition dataset...")

    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")

    # partition the data into training and testing splits using 90% of the data for
    # training and remaining 10% for testing

    split = train_test_split(data, targets, filenames, test_size=0.20, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainTargets, testTargets) = split[2:4]
    (trainFilenames, testFilenames) = split[4:]

    # print(testFilenames)

    pd.Series(testFilenames).to_csv("./test.csv")

    return trainImages, testImages, trainTargets, testTargets, trainFilenames, testFilenames
