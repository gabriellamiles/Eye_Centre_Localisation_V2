"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os
import cv2

import pandas as pd
import numpy as np

from tensorflow.keras.utils import load_img

def load_csv_files_from_directory(directory):

    list_of_df = []

    for i in os.listdir(directory):
        tmp_filepath = os.path.join(directory, i)
        tmp_df = pd.read_csv(tmp_filepath)[["filename", "x", "y"]]
        list_of_df.append(tmp_df)

    return list_of_df

def create_video(df, img_folder, save_folder):

    video_title = df.iloc[0, 0].split("/")[0] + "_" + df.iloc[0, 0].split("/")[1] + ".avi"
    out = cv2.VideoWriter(os.path.join(save_folder, video_title), cv2.VideoWriter_fourcc('M','J','P','G'), 20, (274,274))

    for row in range(df.shape[0]):
        print(str(row) + "/" + str(df.shape[0]))
        filename = str(df["filename"][row])
        img_filepath = os.path.join(img_folder, filename)

        pil_image = load_img(img_filepath)
        open_cv_image = np.array(pil_image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

        h, w, c = open_cv_image.shape

        target_w, target_h = 274, 274
        diff_w, diff_h = target_w-w, target_h-h

        color = [0, 0, 0]
        top, bottom, left, right = [0,0,0,0]
        left, right = [int(diff_w/2), int(diff_w/2)]
        top, bottom = [int(diff_h/2), int(diff_h/2)]

        if diff_w%2 == 1:
            right += 1

        if diff_h%2 == 1:
            bottom += 1


        img_with_border = cv2.copyMakeBorder(open_cv_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # eye centre coordinates
        x = int(df["x"][row])
        y = int(df["y"][row])

        # plot on image
        cv2.circle(img_with_border, (x, y), 4, (255,0,0), -1)

        # cv2.imshow("im", img_with_border)
        # cv2.waitKey(0)
        # write to video
        out.write(img_with_border)

    out.release()

if __name__ == '__main__':

    # initialise key filepaths
    model_directory = "20230208_175022"
    root_folder = os.getcwd()
    predictions_folder = os.path.join(root_folder, "models", model_directory, "predictions")
    img_folder = os.path.join(root_folder, "data", "processed", "mnt", "cropped_eye_imgs")
    save_folder = os.path.join(root_folder, "src", "visualisation", "prediction_videos")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # get predictions
    print("[INFO] Obtaining predictions...")
    list_of_predictions_df = load_csv_files_from_directory(predictions_folder)

    print("[INFO] Creating videos...")
    count = 0
    for df in list_of_predictions_df:
        print("Progress: " + str(count) + "/" + str(len(list_of_predictions_df)))
        create_video(df, img_folder, save_folder)
        count += 1
