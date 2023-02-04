
import os

import pandas as pd


def retrieve_csv_filepaths_from_directory(directory):

    csv_filepaths = []

    for i in os.listdir(directory):
        if i[-4:]==".csv":
            tmp = os.path.join(directory, i)
            csv_filepaths.append(tmp)

    return csv_filepaths

def compare_directory_for_common_participants(list_of_filepaths, directory, replacement=(None, None)):

    updated_list_of_filepaths = []

    for filepath in list_of_filepaths:
        comparison_filepath = filepath.replace(replacement[0], replacement[1])

        if os.path.exists(comparison_filepath):
            updated_list_of_filepaths.append(filepath)


    return updated_list_of_filepaths

def load_csv_files_from_list(list_of_filepaths):

    list_of_df = []

    for filepath in list_of_filepaths:
        tmp_df = pd.read_csv(filepath)
        list_of_df.append(tmp_df)

    return list_of_df

def combine_dataframes(updated_eye_centre_filepaths, bounding_box_folder, save_folder):

    for eye_centre_filepath in updated_eye_centre_filepaths:
        # get corresponding bounding box filepath
        bounding_box_filepath = eye_centre_filepath.replace("centres", "bounding_boxes")
        # load eye centre and corresponding bounding box dfs
        eye_centre_df = pd.read_csv(eye_centre_filepath)
        bounding_box_df = pd.read_csv(bounding_box_filepath)

        combined_list = []
        count = 0
        print(eye_centre_filepath)
        print(eye_centre_df.head())

        for row in eye_centre_df["filename"]:

            corresponding_index = bounding_box_df.index[bounding_box_df["filename"]==row]

            # steal relevant info from bounding box df
            LE_left = int(bounding_box_df["LE_left"][corresponding_index])
            LE_top = int(bounding_box_df["LE_top"][corresponding_index])
            LE_right = int(bounding_box_df["LE_right"][corresponding_index])
            LE_bottom = int(bounding_box_df["LE_bottom"][corresponding_index])
            RE_left = int(bounding_box_df["RE_left"][corresponding_index])
            RE_top = int(bounding_box_df["RE_top"][corresponding_index])
            RE_right = int(bounding_box_df["RE_right"][corresponding_index])
            RE_bottom = int(bounding_box_df["RE_bottom"][corresponding_index])

            # steal revelant info from centres df
            lx = int(eye_centre_df["lx"][count])
            ly = int(eye_centre_df["ly"][count])
            rx = int(eye_centre_df["rx"][count])
            ry = int(eye_centre_df["ry"][count])

            combined_list.append([row, LE_left, LE_top, LE_right, LE_bottom, RE_left, RE_top, RE_right, RE_bottom, lx, ly, rx, ry])

            count += 1

        combined_df = pd.DataFrame(combined_list, columns=["filename", "LE_left", "LE_top", "LE_right", "LE_bottom", "RE_left", "RE_top", "RE_right", "RE_bottom", "lx", "ly", "rx", "ry"])

        save_under = eye_centre_filepath.split("/")[-1]
        save_under = os.path.join(save_folder, save_under)

        combined_df.to_csv(save_under)

if __name__ == '__main__':

    # retrieve filepath for directories containing bounding box and eye centre coordinates, respectively
    bounding_box_folder = os.path.join(os.getcwd(), "data", "raw", "bounding_boxes")
    eye_centre_folder = os.path.join(os.getcwd(), "data", "raw", "centres")
    save_folder = os.path.join(os.getcwd(), "data", "processed", "combined_labels")# folder for storing combined dataframes

    # obtain lists of all eye centre, and bounding box label csv files
    eye_centre_filepaths = retrieve_csv_filepaths_from_directory(eye_centre_folder)

    # compare list for similarities
    updated_eye_centre_filepaths = compare_directory_for_common_participants(eye_centre_filepaths, bounding_box_folder, replacement=("centres","bounding_boxes"))

    # load dataframes of common participants
    eye_centre_dfs = load_csv_files_from_list(updated_eye_centre_filepaths)

    # build new dataframe of existing eye centre data + corresponding dataframe for bounding boxes with relative coordinates in addition
    combined_ec_bb_df = combine_dataframes(updated_eye_centre_filepaths, bounding_box_folder, save_folder)
