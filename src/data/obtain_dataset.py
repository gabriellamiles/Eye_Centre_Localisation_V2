"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os

import pandas as pd

import config
from dataset_utils import retrieve_csv_filepaths_from_directory


class eyeCentreData:
    def __init__(
            self, 
            eye_centre_folders = None, 
            columns = None,
            bounding_columns = None
            ):
        
        if eye_centre_folders is not None:
            self.eye_centre_folders = eye_centre_folders
        else:
            self.eye_centre_folders = None

        if columns is not None:
            self.columns = columns
        else:
            self.columns = None

        if bounding_columns is not None:
            self.bounding_columns = bounding_columns
        else:
            self.bounding_columns = None
    

    def load_filepaths(self, list_of_folders=None):
        """
        Retrieve all filepaths as list of lists, where individual list represents the filepaths in each individual directory.

        """

        self.data_to_combine = []
        # load filepaths in directories of interest

        for directory in list_of_folders:
            self.data_to_combine.append(retrieve_csv_filepaths_from_directory(directory))

        
    def compare_directory_for_common_participants(self, list_of_filepaths, replacement=(None, None)):

        """
        Function takes in list_of_filepaths which contains all files within a gven directory as a sublist

        replacement = tuple of values containing how to change filepaths to get stuff to work
        """

        # initialise variables to store lists of matched and unmatched data
        self.list_of_matched_data = []
        self.list_of_unmatched_data = []

        for i in range(len(list_of_filepaths)):

            # compare filepaths
            for filepath in list_of_filepaths[i]:

                # if this is the first directory being accessed
                if i == 0:

                    for j in range(1, len(list_of_filepaths)):

                        if filepath.replace(replacement[0], replacement[j]) in list_of_filepaths[i+j]:
                            self.list_of_matched_data.append(filepath)
                        else:
                            self.list_of_unmatched_data.append(filepath)

                else:

                    for filepath in list_of_filepaths[i]:

                        if filepath.replace(replacement[i], replacement[0]) in self.list_of_matched_data:
                            continue
                        else:
                            self.list_of_unmatched_data.append(filepath)

        print(len(self.list_of_matched_data), len(self.list_of_unmatched_data))


    def save_unmatched_data(self, replacement_items=["centres", "labels_eme2"], to_replace="combined_centres"):

        """ Pass columns as list: ["filename", "lx"...]"""

        for filepath in self.list_of_unmatched_data:

            tmp_df = pd.read_csv(filepath)[self.columns]

            filename = filepath.rsplit("/",1)[-1]

            path = "."
            if replacement_items[0] in filepath:
                path = filepath.rsplit("/", 1)[0].replace(replacement_items[0], to_replace)
            elif replacement_items[1] in filepath:
                path = filepath.rsplit("/", 1)[0].replace(replacement_items[1], to_replace)

            tmp_df.to_csv(os.path.join(path, filename))

    def save_matched_data(self, replacement_items=["centres", "labels_eme2"], to_replace="combined_centres"):

        for filepath in self.list_of_matched_data:
            corresponding_filepath = filepath.replace(replacement_items[0], replacement_items[1])

            # load both dataframes
            orig_df = pd.read_csv(filepath)[self.columns]
            new_df = pd.read_csv(corresponding_filepath)[self.columns]

            # concatenate dataframes, remove duplicate rows, and reset index
            combined_df = pd.concat([new_df, orig_df])
            combined_df = combined_df.drop_duplicates(subset='filename', keep="first").reset_index(drop=True)

            if replacement_items[0] in filepath:
                saveUnder = filepath.replace(replacement_items[0], to_replace)
            elif replacement_items[1] in filepath:
                saveUnder = filepath.replace(replacement_items[1], to_replace)
            
            combined_df.to_csv(saveUnder)

    def save_data_as_csv(self):
        self.save_unmatched_data()
        self.save_matched_data()

    def save_final_dataset(self, save_folder=config.final_data_folder, replacement=("combined_centres","bounding_boxes")):

        for filepath in self.list_of_matched_data:

            # load dataframes
            centre_df = pd.read_csv(filepath)[self.columns]
            box_df = pd.read_csv(filepath.replace(replacement[0], replacement[1]))[self.bounding_columns]

            # combine dataframes
            combined_list = []
            count = 0

            for row in centre_df["filename"]:

                corresponding_index = box_df.index[box_df["filename"]==row]

                # steal relevant info from bounding box df
                LE_left = int(box_df["LE_left"][corresponding_index])
                LE_top = int(box_df["LE_top"][corresponding_index])
                LE_right = int(box_df["LE_right"][corresponding_index])
                LE_bottom = int(box_df["LE_bottom"][corresponding_index])
                RE_left = int(box_df["RE_left"][corresponding_index])
                RE_top = int(box_df["RE_top"][corresponding_index])
                RE_right = int(box_df["RE_right"][corresponding_index])
                RE_bottom = int(box_df["RE_bottom"][corresponding_index])

                # steal revelant info from centres df
                lx = int(centre_df["lx"][count])
                ly = int(centre_df["ly"][count])
                rx = int(centre_df["rx"][count])
                ry = int(centre_df["ry"][count])

                # calculate relative eye centres
                relative_lx = lx-LE_left
                relative_ly = ly-LE_top
                relative_rx = rx-RE_left
                relative_ry = ry-RE_top

                combined_list.append([row, LE_left, LE_top, LE_right, LE_bottom, RE_left, RE_top, RE_right, RE_bottom, lx, ly, rx, ry, relative_lx, relative_ly, relative_rx, relative_ry])

                count += 1 # increment

            combined_df = pd.DataFrame(combined_list, columns=["filename", "LE_left", "LE_top", "LE_right", "LE_bottom", "RE_left", "RE_top", "RE_right", "RE_bottom", "lx", "ly", "rx", "ry", "relative_lx", "relative_ly", "relative_rx", "relative_ry"])

            save_under = filepath.split("/")[-1]

            save_under = os.path.join(save_folder, save_under)

            combined_df.to_csv(save_under)
            

if __name__ == '__main__':

    dataset = eyeCentreData()
    dataset.eye_centre_folders = config.eye_centre_folders_raw
    dataset.columns = config.eye_centre_columns
    dataset.bounding_columns = config.bounding_columns

    # dataset.load_filepaths(dataset.eye_centre_folders)
    # dataset.compare_directory_for_common_participants(dataset.data_to_combine, replacement=("centres", "labels_eme2"))

    # dataset.save_data_as_csv() # save combined labels

    # now sort out comparing data from combined labels to bounding box, and saving this data in processed/combined_labels
    dataset.load_filepaths(list_of_folders=[config.combined_eye_centre_folder, config.bounding_box_folder])
    dataset.compare_directory_for_common_participants(dataset.data_to_combine, replacement=("combined_centres","bounding_boxes"))
    dataset.save_final_dataset(replacement=("combined_centres","bounding_boxes"))