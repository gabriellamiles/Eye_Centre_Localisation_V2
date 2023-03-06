"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os

import pandas as pd

def get_corresponding_filepaths(centres_filepaths, bounding_box_filepaths):

    updated_centres_filepaths, updated_bounding_box_filepaths = [], []

    for filepath in centres_filepaths:
        tmp_filepath = filepath.replace("models/20230210_173438/predictions/left_eye", "data/raw/bounding_boxes")
        if os.path.exists(tmp_filepath):
            updated_centres_filepaths.append(filepath)
            updated_bounding_box_filepaths.append(tmp_filepath)


    return updated_centres_filepaths, updated_bounding_box_filepaths

def load_csv_files_from_filepaths(list_of_filepaths, columns):

    list_of_df = []

    for filepath in list_of_filepaths:
        tmp_df = pd.read_csv(filepath)[columns]
        list_of_df.append(tmp_df)

    return list_of_df

def get_absolute_coordinates(list_of_centre_df, list_of_bounding_box_df, size, save_folder):

    for i in range(len(list_of_centre_df)):

        centre_df = list_of_centre_df[i]
        bounding_box_df = list_of_bounding_box_df[i]

        absolute_centres = []

        eye = centre_df["filename"][0].split("/")[0]
        participant = centre_df["filename"][0].split("/")[1]
        trial = centre_df["filename"][0].split("/")[2]

        for row in range(centre_df.shape[0]):

            centre_filename = centre_df["filename"][row]
            check_centre_filename = centre_filename.split("_eye/")[-1]
            bounding_box_filename = bounding_box_df["filename"][row]

            if check_centre_filename != bounding_box_filename:
                continue

            left, right, top, bottom = 0, 0, 0, 0
            if "left_eye" in centre_filename:

                left = bounding_box_df["LE_left"][row]
                right = bounding_box_df["LE_right"][row]
                top = bounding_box_df["LE_top"][row]
                bottom = bounding_box_df["LE_bottom"][row]

            elif "right_eye" in centre_filename:

                left = bounding_box_df["RE_left"][row]
                right = bounding_box_df["RE_right"][row]
                top = bounding_box_df["RE_top"][row]
                bottom = bounding_box_df["RE_bottom"][row]

            width = right - left
            diff_width = size[0] - width
            diff_in_x = int(diff_width/2)

            height = bottom-top
            diff_height = size[1] - height
            diff_in_y = int(diff_height/2)

            x = centre_df["x"][row]
            y = centre_df["y"][row]

            print(width, diff_width, height, diff_height)
            print(x, y)

            new_x = x - diff_in_x # relative to bounding box coordinates
            new_y = y - diff_in_y
            print(new_x, new_y)

            abs_x = new_x + left # relative to absolute photo coordinates
            abs_y = new_y + top
            print(abs_x, abs_y)
            absolute_centres.append([centre_filename, abs_x, abs_y])

        absolute_centres_df = pd.DataFrame(absolute_centres, columns=["filename", "abs_x", "abs_y"])
        csv_save_name = participant + "_" + trial + ".csv"
        save_filepath = os.path.join(save_folder, csv_save_name)

        absolute_centres_df.to_csv(save_filepath)


if __name__ == '__main__':

    #initialise keyfilepaths
    root_folder = os.getcwd()
    centres_folder = os.path.join(root_folder, "models", "20230210_173438", "predictions", "left_eye")
    bounding_box_folder = os.path.join(root_folder, "data", "raw", "bounding_boxes")
    save_folder = os.path.join(root_folder, "data", "processed", "centre_coordinates", "left_eye")

    # initialise key variables
    size = (224,224)

    # retrieve label filepaths
    centres_filepaths = [os.path.join(centres_folder, i) for i in os.listdir(centres_folder)]
    bounding_box_filepaths = [os.path.join(bounding_box_folder, i) for i in os.listdir(bounding_box_folder)]
    print(len(centres_filepaths), len(bounding_box_filepaths))

    # retrieve labels for which both centres and bounding box data exist
    centres_filepaths, bounding_box_filepaths = get_corresponding_filepaths(centres_filepaths, bounding_box_filepaths)
    print(len(centres_filepaths), len(bounding_box_filepaths))
    print(centres_filepaths[0], bounding_box_filepaths[0])


    # load data
    list_of_centre_df = load_csv_files_from_filepaths(centres_filepaths, columns=["filename", "x", "y"])
    list_of_bounding_box_df = load_csv_files_from_filepaths(bounding_box_filepaths, columns=["filename", "LE_left", "LE_top", "LE_right", "LE_bottom", "RE_left", "RE_top", "RE_right", "RE_bottom"])
    print(len(list_of_centre_df), len(list_of_bounding_box_df))

    # combine data correctly
    get_absolute_coordinates(list_of_centre_df, list_of_bounding_box_df, size, save_folder)
