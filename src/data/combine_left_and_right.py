"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os
import pandas as pd

def load_csv_files_from_filepaths(list_of_filepaths):

    list_of_df = []

    for filepath in list_of_filepaths:
        df = pd.read_csv(filepath)[["filename", "abs_x", "abs_y"]]
        list_of_df.append(df)

    return list_of_df

def combine_dataframes(corresponding_left_df, corresponding_right_df):

    list_of_combined_df = []

    for i in range(len(corresponding_left_df)):


        left_df = corresponding_left_df[i]
        left_df.rename(columns={"abs_x": "lx", "abs_y": "ly"}, inplace=True)
        right_df = corresponding_right_df[i]
        right_df.rename(columns={"filename": "right_filename", "abs_x": "rx", "abs_y": "ry"}, inplace=True)

        both_df = pd.concat([left_df, right_df], axis=1)
        # print(both_df.head())
        both_df = both_df[["filename", "lx", "ly", "rx", "ry"]]
        both_df["filename"] = both_df["filename"].str.replace("left_eye/", "")
        # print(both_df.head())
        # print(left_df.shape, right_df.shape, both_df.shape)

        list_of_combined_df.append(both_df)

    return list_of_combined_df


def get_corresponding_dataframes(left_eye_folder, right_eye_folder):

    left_eye_filepaths = [os.path.join(left_eye_folder, i) for i in os.listdir(left_eye_folder)]

    corresponding_left_filepaths = []
    corresponding_right_filepaths = []

    for filepath in left_eye_filepaths:
        new_filepath = filepath.replace("left", "right")

        if os.path.exists(new_filepath):
            corresponding_left_filepaths.append(filepath)
            corresponding_right_filepaths.append(new_filepath)

    # print(len(corresponding_left_filepaths), len(corresponding_right_filepaths))
    corresponding_left_df = load_csv_files_from_filepaths(corresponding_left_filepaths)
    corresponding_right_df = load_csv_files_from_filepaths(corresponding_right_filepaths)
    # print(corresponding_left_df[0].shape, corresponding_right_df[0].shape)

    corresponding_df = combine_dataframes(corresponding_left_df, corresponding_right_df)

    return corresponding_df

def organise_dataframe_consecutive(list_of_df):

    list_of_consecutive_df = []

    for df in list_of_df:

        df = df.sort_values(by=["filename"])
        df = df.reset_index(drop=True)

        list_of_consecutive_df.append(df)

    return list_of_consecutive_df

if __name__ == '__main__':

    # initialise key filepaths
    root_folder = os.getcwd()
    left_eye_folder = os.path.join(root_folder, "data", "processed", "centre_coordinates", "left_eye")
    right_eye_folder = os.path.join(root_folder, "data", "processed", "centre_coordinates", "right_eye")
    save_folder = os.path.join(root_folder, "data", "processed", "centre_coordinates", "both_eyes")

    # match filenames in left eye folder to right  eye folder and get filepaths for
    # ones where both left and rigth eye exist
    both_eyes_df = get_corresponding_dataframes(left_eye_folder, right_eye_folder)
    both_eyes_df = organise_dataframe_consecutive(both_eyes_df)

    # save dataframes
    for df in both_eyes_df:
        participant = df["filename"][0].split("/")[0]
        trial = df["filename"][0].split("/")[1]
        csv_save_name = participant + "_" + trial + ".csv"
        save_under = os.path.join(save_folder, csv_save_name)
        df.to_csv(save_under)
