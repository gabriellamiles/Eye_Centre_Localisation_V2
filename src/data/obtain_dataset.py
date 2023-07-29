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
   
    def compare_directory_for_common_participants(self, replacement=(None, None)):

        """
        Function takes in list_of_filepaths which contains all files within a gven directory as a sublist

        replacement = tuple of values containing how to change filepaths to get stuff to work
        """

        # # initialise variables to store lists of matched and unmatched data
        self.list_of_matched_data = []
        self.list_of_unmatched_data = []

        grp_1 = self.data_to_combine[0]
        grp_2 = self.data_to_combine[1]

        for filepath in grp_1:
            tmp_filepath = filepath.replace(replacement[0], replacement[1])

            if tmp_filepath in grp_2:
                self.list_of_matched_data.append(tmp_filepath)
            else:
                self.list_of_unmatched_data.append(tmp_filepath)

        for filepath in grp_2:
            
            if filepath in self.list_of_matched_data:
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

            print(self.columns)
            print(self.bounding_columns)
            print(filepath)

            # load dataframes
            centre_df = pd.read_csv(filepath.replace("predictions", "combined_centres"))[self.columns]
            box_df = pd.read_csv(filepath)[self.bounding_columns]

            # combine dataframes
            combined_list = []
            count = 0

            # print(centre_df.head())

            for row in centre_df["filename"]:

                corresponding_index = box_df.index[box_df["filename"]==row]
                print(corresponding_index)

                print(centre_df.head())
                print(box_df.head())

                # steal relevant info from bounding box df
                LE_left = int(box_df["resized_LE_left"][corresponding_index])
                LE_top = int(box_df["resized_LE_top"][corresponding_index])
                LE_right = int(box_df["resized_LE_right"][corresponding_index])
                LE_bottom = int(box_df["resized_LE_bottom"][corresponding_index])
                RE_left = int(box_df["resized_RE_left"][corresponding_index])
                RE_top = int(box_df["resized_RE_top"][corresponding_index])
                RE_right = int(box_df["resized_RE_right"][corresponding_index])
                RE_bottom = int(box_df["resized_RE_bottom"][corresponding_index])

                # steal revelant info from centres df
                lx = int(centre_df["lx"][count])
                ly = int(centre_df["ly"][count])
                rx = int(centre_df["rx"][count])
                ry = int(centre_df["ry"][count])

                # calculate relative eye centres
                relative_lx = lx-LE_left
                if relative_lx < 0: 
                    relative_lx = 0

                relative_ly = ly-LE_top
                if relative_ly < 0:
                    relative_ly = 0
                
                relative_rx = rx-RE_left
                if relative_rx < 0:
                    relative_rx = 0

                relative_ry = ry-RE_top
                if relative_ry < 0:
                    relative_ry = 0

                combined_list.append([row, LE_left, LE_top, LE_right, LE_bottom, RE_left, RE_top, RE_right, RE_bottom, lx, ly, rx, ry, relative_lx, relative_ly, relative_rx, relative_ry])

                count += 1 # increment

            combined_df = pd.DataFrame(combined_list, columns=["filename", "relative_LE_left", "relative_LE_top", "relative_LE_right", "relative_LE_bottom", "relative_RE_left", "relative_RE_top", "relative_RE_right", "relative_RE_bottom", "lx", "ly", "rx", "ry", "relative_lx", "relative_ly", "relative_rx", "relative_ry"])

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
    dataset.compare_directory_for_common_participants(replacement=("combined_centres","predictions"))
    dataset.save_final_dataset(replacement=("final_dataset","bounding_boxes"))