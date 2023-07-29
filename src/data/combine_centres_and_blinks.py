"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os
import pandas as pd

def get_filepaths_from_directory(directory):

    all_filepaths = [os.path.join(directory, i) for i in os.listdir(directory)]

    return all_filepaths

def get_matching_filepaths(list_1, list_2):

    # to store matching results

    matching_filenames = []

    for i in list_1:    
        participant_trial = os.path.basename(i)

        for j in list_2:
            
            if participant_trial in j:
                
                matching_filenames.append(j)
                break

    return matching_filenames

def update_results_with_blinks(matched_filenames, eye_region_folder):
    

    for participant_trial in matched_filenames:

        filename = os.path.join(eye_region_folder, os.path.basename(participant_trial))
        print(filename)
        
        # ecl_df = pd.read_csv().sort_values(by=["filename"], inplace=False)
        # blink_df = pd.read_csv(participant_trial).sort_values(by=["filename"], inplace=False)

        # print(ecl_df.head())
        # print(blink_df.head())

        break


if __name__ == '__main__':
    
    # initialise key filepaths
    root_folder = os.getcwd()
    eye_region_folder = os.path.join(root_folder, "output", "bb_predictions").replace("Eye_Centre_Localisation_V2", "Eye_Region_Detection")
    blink_folder = os.path.join(root_folder, "output").replace("Eye_Centre_Localisation_V2", "Blink_Detector")

    eye_region_filepaths = get_filepaths_from_directory(eye_region_folder)
    blink_filepaths = get_filepaths_from_directory(blink_folder)
    # print(len(eye_region_filepaths), len(blink_filepaths))

    matching_filepaths =  get_matching_filepaths(eye_region_filepaths, blink_filepaths)
    # print(matching_filepaths)

    update_results_with_blinks(matching_filepaths, eye_region_folder)

