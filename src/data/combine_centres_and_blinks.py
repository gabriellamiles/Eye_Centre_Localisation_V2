"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os
import pandas as pd

from itertools import groupby
from operator import itemgetter

def get_filepaths_from_directory(directory):

    all_filepaths = [os.path.join(directory, i) for i in os.listdir(directory)]

    return all_filepaths

def get_matching_filepaths(list_1, list_2):

    # to store matching results

    blink_matching_filenames = []
    centre_matching_filenames = []

    for i in list_1:    
        participant_trial = os.path.basename(i)

        for j in list_2:
            
            if participant_trial in j:
                
                blink_matching_filenames.append(j)
                centre_matching_filenames.append(i)
                break

    return blink_matching_filenames, centre_matching_filenames

def update_results_with_blinks(matched_filenames, eye_region_folder):
    

    for participant_trial in matched_filenames:

        filename = os.path.join(eye_region_folder, os.path.basename(participant_trial))
        print(filename)
        
        # ecl_df = pd.read_csv().sort_values(by=["filename"], inplace=False)
        # blink_df = pd.read_csv(participant_trial).sort_values(by=["filename"], inplace=False)

        # print(ecl_df.head())
        # print(blink_df.head())

        break


def expand_and_preprocess_blinks(blink_matching_filepaths, centre_matching_filenames):


    all_combined_df = []
    count=0
    for filepath in blink_matching_filepaths:

        # check that the order of lists matches
        if os.path.basename(filepath) == os.path.basename(centre_matching_filenames[count]):
            pass
        else:
            print("Doesn't match")
            continue
        
        # load files as dfs
        blink_tmp_df = pd.read_csv(filepath)[["filepath", "blink"]]
        centres_tmp_df = pd.read_csv(centre_matching_filenames[count])[['filename', 'abs_lx', 'abs_ly', 'abs_rx', 'abs_ry']]

        # sort blink and centres folder by order
        blink_tmp_df = blink_tmp_df.sort_values(by=["filepath"])
        blink_tmp_df["blink"] = blink_tmp_df["blink"].round(0).astype(int)
        centres_tmp_df = centres_tmp_df.sort_values(by=["filename"])

        # extract frames identified as blinks
        # blink_tmp_df = blink_tmp_df[blink_tmp_df["blink"]==0]
        # blink_tmp_df["filepath"] = blink_tmp_df["filepath"].str.replace("_left", "")

        # print(centres_tmp_df[["filename"]].head())
        # print(blink_tmp_df.head())

        # combine blinks and centres
        combined_df = centres_tmp_df.merge(blink_tmp_df,how='left', left_on='filename', right_on='filepath')
        print(combined_df.head())
        # blink_indices = combined_df.index[combined_df["blink"]==0].tolist()

        # full_blink_indices = []
        # for blink_frame_number in blink_indices:

        #     # add buffer of one frames before 
        #     buffered_blink = [item for item in range(blink_frame_number-1, blink_frame_number)]
        #     for item in buffered_blink:
        #         full_blink_indices.append(item)

        # # keep only unique items
        # full_blink_indices = set(full_blink_indices)
        # full_blink_indices = [item for item in full_blink_indices if item >= 0]

        # # expand blinks
        # combined_df.loc[full_blink_indices, "blink"] = 0
        

        # set nan blinks to zero
        combined_df["blink"].fillna(1, inplace=True)
        combined_df.drop(columns=["filepath"], inplace=True)
    #     combined_df.to_csv("test.csv")


        all_combined_df.append(combined_df)

        count+=1

    return all_combined_df


def update_traces(all_combined_df):

    for df in all_combined_df:

        # extract participant and trial number for saving purposes
        participant = df["filename"].iloc[0].split("/")[0]
        trial = df["filename"].iloc[0].split("/")[1]

        # find blink locations
        # blink_indices = df.index[df["blink"]==1].tolist()

        # for k, g in groupby(enumerate(blink_indices), lambda i_x: i_x[0] - i_x[1]):
        #     # extract blink segment
        #     blink_segment = list(map(itemgetter(1), g))

        #     # get index for last open eye
        #     last_open_eye_frame = blink_segment[0]-1
        #     if last_open_eye_frame < 0:
        #         last_open_eye_frame = blink_segment[0]+1

        #     # get values of last open eye
        #     lx_last_open = df["abs_lx"].loc[last_open_eye_frame]
        #     ly_last_open = df["abs_ly"].loc[last_open_eye_frame]
        #     rx_last_open = df["abs_rx"].loc[last_open_eye_frame]
        #     ry_last_open = df["abs_ry"].loc[last_open_eye_frame]


        #     # set values to that of last open eye
        #     df.loc[blink_segment, "abs_lx"] = lx_last_open
        #     df.loc[blink_segment, "abs_ly"] = ly_last_open
        #     df.loc[blink_segment, "abs_rx"] = rx_last_open
        #     df.loc[blink_segment, "abs_ry"] = ry_last_open

        df.columns = ["filename", "abs_lx", "abs_ly", "abs_rx", "abs_ry", "open_eyes"]
        save_name = os.path.join(os.getcwd(), "output", "fin_eye_movement_traces_inc_blinks", participant + "_" + trial + ".csv")
        print(save_name)
        df.to_csv(save_name)

if __name__ == '__main__':
    
    # initialise key filepaths
    root_folder = os.getcwd()
    eye_region_folder = os.path.join(root_folder, "output", "eye_movement_traces_test")
    blink_folder = os.path.join(root_folder, "blink_output", "fin_preds").replace("Eye_Centre_Localisation_V2", "Blink_Detector")

    eye_region_filepaths = sorted(get_filepaths_from_directory(eye_region_folder))
    blink_filepaths = sorted(get_filepaths_from_directory(blink_folder))
    # print(len(eye_region_filepaths), len(blink_filepaths))

    blink_matching_filepaths, centre_matching_filenames =  get_matching_filepaths(eye_region_filepaths, blink_filepaths)
    print(len(blink_matching_filepaths), len(centre_matching_filenames))

    all_combined_df = expand_and_preprocess_blinks(blink_matching_filepaths, centre_matching_filenames)

    update_traces(all_combined_df)


