import os

import pandas as pd

from pathlib import Path

def load_csv_files():

    pass

def match_participants(file_list_1, file_list_2):

    matching_filepaths = []

    for filename in file_list_1:
        participant_trial = os.path.basename(filename)

        for filename_2 in file_list_2:
            if participant_trial == os.path.basename(filename_2):
                matching_filepaths.append(filename_2)
                break

    return matching_filepaths

def combine_dataframes():

    pass

def set_blinks_to_zero():

    pass

def get_list_of_files(directory):

    files = [os.path.join(directory, i) for i in os.listdir(directory)]

    return files

if __name__ == '__main__':

    # initialise key filepaths
    root_folder = os.getcwd()
    bounding_box_folder = os.path.join(root_folder, "bb_predictions")
    centres_folder = os.path.join(root_folder, "output", "patch_predictions")
    save_folder = os.path.join(root_folder, "output", "eye_movement_traces")

    # get list of files
    bounding_box_data = get_list_of_files(bounding_box_folder)
    eye_centre_data = get_list_of_files(centres_folder)

    # get list of matching participants
    matched_data = match_participants(bounding_box_data, eye_centre_data)

    print(len(bounding_box_data), len(eye_centre_data), len(matched_data))

    contains_nan = []

    # for given participants load csv files
    for filename in matched_data:
        print(filename)

        # load all three dataframes
        try:
            eye_centres_df = pd.read_csv(filename)[["filename", "pred_x", "pred_y"]]
            # left_eye_df = pd.read_csv(filename.replace("right", "left"))[["filename", "x", "y"]]
            bounding_box_df = pd.read_csv(os.path.join(bounding_box_folder, os.path.basename(filename)))[["filename", "resized_LE_left", "resized_LE_top", "resized_LE_right", "resized_LE_bottom", "resized_RE_left", "resized_RE_top", "resized_RE_right", "resized_RE_bottom"]]
        except:
            print(f"{os.path.basename(filename)}: couldn't load all three csv files.")

        # sort order and get rid of prefix for right and left eyes
        right_eye_df = eye_centres_df[eye_centres_df["filename"].str.contains("right")]
        right_eye_df['filename'] = right_eye_df['filename'].str.replace('_right', '', regex=True)

        left_eye_df = eye_centres_df[eye_centres_df["filename"].str.contains("left")]
        left_eye_df['filename'] = left_eye_df['filename'].str.replace('_left', '', regex=True) 

        right_eye_df = right_eye_df.sort_values(by=["filename"], inplace=False, ignore_index=True)
        left_eye_df = left_eye_df.sort_values(by=["filename"], inplace=False, ignore_index=True)
        bounding_box_df = bounding_box_df.sort_values(by=["filename"], inplace=False, ignore_index=True)

        if right_eye_df.shape[0] != left_eye_df.shape[0]:
            continue


        # print(right_eye_df.head())
        # print(left_eye_df.head())
        # print(bounding_box_df.head())

    #     # calculations for correct relative x,y coordinates
    #     right_eye_df[["x", "y"]] = (right_eye_df[["x", "y"]]/224)*324
    #     left_eye_df[["x", "y"]] = (left_eye_df[["x", "y"]]/224)*324

    #     # check largest bounding box size
        # bounding_box_df['l_width'] = bounding_box_df['resized_LE_right'] - bounding_box_df['resized_LE_left']
        # bounding_box_df['r_width'] = bounding_box_df['resized_RE_right'] - bounding_box_df['resized_RE_left']
        # bounding_box_df['l_height'] = bounding_box_df['resized_LE_bottom'] - bounding_box_df['resized_LE_top']
        # bounding_box_df['r_height'] = bounding_box_df['resized_RE_bottom'] - bounding_box_df['resized_RE_top']
        # cols = bounding_box_df.columns.to_list() + ["rel_lx", "rel_ly", "rel_rx", "rel_ry"]

        # combine dataframes
        # combined_dataframe = pd.concat([bounding_box_df, left_eye_df[["pred_x", "pred_y"]], right_eye_df[["pred_x", "pred_y"]]], axis=1)
        combined_dataframe = pd.concat([left_eye_df, right_eye_df], axis=1)
        # combined_dataframe.columns = cols

        check_nan = combined_dataframe.isnull().values.any()

        df1 = combined_dataframe[combined_dataframe.isna().any(axis=1)]
        df2 = right_eye_df[right_eye_df.isna().any(axis=1)]
        df3 = left_eye_df[left_eye_df.isna().any(axis=1)]
        
        if df1.shape[0]>500:

            print(df1.shape, df2.shape, df3.shape)
            print(left_eye_df.shape, right_eye_df.shape)
            print(df1.head())
            contains_nan.append(filename)


    #     combined_dataframe["rel_lx"] = combined_dataframe["tmp_lx"]-combined_dataframe["cropped_im_left_padding"]
    #     combined_dataframe["rel_ly"] = combined_dataframe["tmp_ly"]-combined_dataframe["cropped_im_left_top_padding"]
    #     combined_dataframe["rel_rx"] = combined_dataframe["tmp_rx"]-combined_dataframe["cropped_im_right_padding"]        
    #     combined_dataframe["rel_ry"] = combined_dataframe["tmp_ry"]-combined_dataframe["cropped_im_right_top_padding"]

    #     combined_dataframe["abs_lx"] = combined_dataframe["rel_lx"] + combined_dataframe["resized_LE_left"]
    #     combined_dataframe["abs_ly"] = combined_dataframe["rel_ly"] + combined_dataframe["resized_LE_top"]
    #     combined_dataframe["abs_rx"] = combined_dataframe["rel_rx"] + combined_dataframe["resized_RE_left"]
    #     combined_dataframe["abs_ry"] = combined_dataframe["rel_ry"] + combined_dataframe["resized_RE_top"]        

    #     # print(combined_dataframe.head())

    #     # print(combined_dataframe.head())
    # #     # preprocess data (i.e. set blinks to zero)

    #     # save data at relevant save location
    #     save_dataframe = combined_dataframe[["filename", "abs_lx", "abs_ly", "abs_rx", "abs_ry"]]
    #     save_dataframe.to_csv(os.path.join(save_folder, os.path.basename(filename)))

    print(len(contains_nan))
    print(contains_nan)


