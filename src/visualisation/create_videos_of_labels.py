import os
import cv2

import pandas as pd


def load_csv_files(list_of_filepaths):

    list_of_df = []

    for filepath in list_of_filepaths:
        tmp_df = pd.read_csv(filepath)[["filename", "LE_left", "LE_top", "LE_right", "LE_bottom", "RE_left", "RE_top", "RE_right", "RE_bottom", "abs_lx", "abs_ly", "abs_rx", "abs_ry"]]
        list_of_df.append(tmp_df)

    return list_of_df

def create_videos(list_of_df, img_folder):

    # intialise video

    for df in list_of_df:

        # extract information to establish what participant and trial number video should be saved under
        saveFolder = os.path.join(os.getcwd(), "src", "visualisation", "labelInspectionVideos")
        video_title = df.iloc[0, 0].split("/")[0] + "_" + df.iloc[0, 0].split("/")[1] + ".avi"
        out = cv2.VideoWriter(os.path.join(saveFolder, video_title), cv2.VideoWriter_fourcc('M','J','P','G'), 20, (960,960))

        for row in range(df.shape[0]):
            filename = df.iloc[row, 0]
            img_filepath = os.path.join(img_folder, filename)
            image = cv2.imread(img_filepath)

            # left eye bounding box coordinates
            LE_left = int(df.iloc[row, 1])
            LE_top = int(df.iloc[row, 2])
            LE_right = int(df.iloc[row, 3])
            LE_bottom = int(df.iloc[row, 4])

            # plot on image
            cv2.rectangle(image, (LE_left, LE_top), (LE_right, LE_bottom), (255,0,0), 4)

            # right eye bounding box coordinates
            RE_left = int(df.iloc[row, 5])
            RE_top = int(df.iloc[row, 6])
            RE_right = int(df.iloc[row, 7])
            RE_bottom = int(df.iloc[row, 8])

            # plot on image
            cv2.rectangle(image, (RE_left, RE_top), (RE_right, RE_bottom), (0,255,0), 4)

            # left eye centre coordinates
            lx = int(df.iloc[row, 9])
            ly = int(df.iloc[row, 10])

            # plot on image
            cv2.circle(image, (lx, ly), 4, (255,0,0), -1)

            # right eye centre coordinates
            rx = int(df.iloc[row, 11])
            ry = int(df.iloc[row, 12])

            # plot on image
            cv2.circle(image, (rx, ry), 4, (0,255,0), -1)

            # write to video
            out.write(image)

        out.release()

if __name__ == '__main__':

    root_folder = os.getcwd()

    # load all csv files in directory and store as list of dataframes
    label_folder = os.path.join(root_folder, "data", "processed", "combined_labels")
    label_filepaths = [os.path.join(label_folder, i) for i in os.listdir(label_folder)]
    all_df = load_csv_files(label_filepaths)

    # establish filepath to image folder
    img_folder = os.path.join(root_folder, "data", "processed", "mnt", "eme2_square_imgs")
    # currently mounted to different folder so temporarily use this one
    img_folder = img_folder.replace("Eye_Centre_Localisation_V2", "Eye_Region_Detection")

    create_videos(all_df, img_folder)
