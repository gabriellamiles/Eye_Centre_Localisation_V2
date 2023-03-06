"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os

import pandas as pd

def retrieve_csv_filepaths_from_directory(directory):

    csv_filepaths = []

    for i in os.listdir(directory):
        if i[-4:]==".csv":
            tmp = os.path.join(directory, i)
            csv_filepaths.append(tmp)

    return csv_filepaths

def combine_data(list_1, list_2):

    no_matching_data = []
    matching_data = []

    # compare filepaths
    for filepath in list_1:

        if filepath.replace("centres", "labels_eme2") in list_2:
            matching_data.append(filepath)
            # match by row
        else:
            no_matching_data.append(filepath)


    for filepath in list_2:
        if filepath.replace("labels_eme2", "centres") in matching_data:
            continue
        else:
            no_matching_data.append(filepath)

    return no_matching_data, matching_data

def copy_across_unmatched_data(no_matching_data):

    for filepath in no_matching_data:

        tmp_df = pd.read_csv(filepath)[["filename", "lx", "ly", "rx", "ry"]]

        print(filepath)
        filename = filepath.rsplit("/",1)[-1]

        path = "."
        if "centres" in filepath:
            path = filepath.rsplit("/", 1)[0].replace("centres", "combined_centres")
        elif "labels_eme2" in filepath:
            path = filepath.rsplit("/", 1)[0].replace("labels_eme2", "combined_centres")

        tmp_df.to_csv(os.path.join(path, filename))


def combine_matched_data(matching_data):

    for filepath in matching_data:

        new_filepath = filepath.replace("labels_eme2", "centres")

        # load both dataframes
        orig_df = pd.read_csv(filepath)[["filename", "lx", "ly", "rx", "ry"]]
        new_df = pd.read_csv(new_filepath)[["filename", "lx", "ly", "rx", "ry"]]

        # concatenate dataframes, then remove duplicate rows
        new_df = pd.concat([new_df, orig_df])
        new_df = new_df.drop_duplicates(subset='filename', keep="first")

        saveUnder = filepath.replace("centres", "combined_centres")
        new_df.to_csv(saveUnder)


if __name__ == '__main__':

    # retrieve filepaths for directories containing eye centre coordinates from all sourcess
    print("[INFO] Initialising key filepaths...")
    root_folder = os.getcwd()
    eye_centre_folder_1 = os.path.join(root_folder, "data", "raw", "centres")
    eye_centre_folder_2 = os.path.join(root_folder, "data", "raw", "labels_eme2")

    # obtain lists of all eye centre files in these directories
    print("[INFO] Retrieve filepaths from directories...")
    eye_centre_filepaths_f1 = retrieve_csv_filepaths_from_directory(eye_centre_folder_1)
    eye_centre_filepaths_f2 = retrieve_csv_filepaths_from_directory(eye_centre_folder_2)

    print("[INFO] Determine dataframes to combine...")
    no_matching_data, matching_data = combine_data(eye_centre_filepaths_f1, eye_centre_filepaths_f2)
    print(str(len(no_matching_data)) + " csv files with no corresponding participant.")
    print(str(len(matching_data)) + " csv files with corresponding participant.")

    print("[INFO] Save dataframes with no corresponding participant to new directory...")
    copy_across_unmatched_data(no_matching_data)

    print("[INFO] Combine dataframes with corresponding participant and save to new directory...")
    combine_matched_data(matching_data)
