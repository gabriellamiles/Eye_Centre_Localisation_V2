import os
import cv2

import pandas as pd
import numpy as np

def load_csv_files_from_directory(directory):

    list_of_csv_files = []
    left_eye_df = []
    right_eye_df = []

    for file in os.listdir(directory):

        filepath = os.path.join(directory, file)

        if filepath[-4:] == ".csv":
            tmp_df = pd.read_csv(filepath)[["filename", "LE_left", "LE_top", "LE_right", "LE_bottom", "RE_left", "RE_top", "RE_right", "RE_bottom"]]
            left_eye_df.append(tmp_df[["filename", "LE_left", "LE_top", "LE_right", "LE_bottom"]]) # extract left eye coordinates
            right_eye_df.append(tmp_df[["filename", "RE_left", "RE_top", "RE_right", "RE_bottom"]]) # extract right eye coordinates
            list_of_csv_files.append(tmp_df)

    return list_of_csv_files, left_eye_df, right_eye_df

def extract_eye_image(list_of_df, img_folder):

    count = 0
    for df in list_of_df: # for each trial

        print("Progress: " + str(count)+"/"+str(len(list_of_df)))

        for row in range(df.shape[0]): # idex for every row in trial

            filepath = df["filename"][row]
            full_im_filepath = os.path.join(img_folder, filepath)

            im = cv2.imread(full_im_filepath)
            left = int(df["RE_left"][row])
            top = int(df["RE_top"][row])
            right = int(df["RE_right"][row])
            bottom = int(df["RE_bottom"][row])


            cropped_im = im[top:bottom, left:right]

            # if row == 0:
            #     cv2.imshow("im", cropped_im)
            #     cv2.waitKey(0)

            save_under = full_im_filepath.replace("eme2_square_imgs", "cropped_eye_imgs/right_eye")
            # save_under = os.path.join(save_under, "right_eye")
            save_directory = save_under.rsplit("/",1)[0]

            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # save image
            cv2.imwrite(save_under, cropped_im)

        count +=1

if __name__ == '__main__':

    # initialise key filepaths
    root_folder = os.getcwd()
    img_folder = os.path.join(root_folder, "data", "processed", "mnt", "eme2_square_imgs")
    bounding_box_labels = os.path.join(root_folder, "data", "raw", "bounding_boxes")

    # obtain bounding box labels
    print("[INFO] Obtaining labels...")
    bounding_box_df, left_eye_df, right_eye_df = load_csv_files_from_directory(bounding_box_labels)
    extract_eye_image(right_eye_df, img_folder)
