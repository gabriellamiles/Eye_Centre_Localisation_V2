"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os
import cv2
import pandas as pd

def get_csv_filepaths_from_directory(directory):

    list_of_filepaths = []

    for file in os.listdir(directory):
        if file[-4:] == ".csv":
            list_of_filepaths.append(os.path.join(directory, file))

    return list_of_filepaths

def get_corresponding_labels(labels_folder, img_folder):

    corresponding_filepaths = []

    labels_filepaths = get_csv_filepaths_from_directory(labels_folder)

    for filepath in labels_filepaths:
        tmp_filepath = filepath.replace(".csv", "")
        tmp_filepath = tmp_filepath.replace("combined_labels", "mnt/cropped_eye_imgs")
        tmp_filepath = tmp_filepath.rsplit("_",1)

        full_filepath = ''
        for item in tmp_filepath:
            full_filepath = os.path.join(full_filepath, item)

        if not os.path.exists(full_filepath):
            # print(full_filepath)
            continue

        corresponding_filepaths.append(filepath)

    return corresponding_filepaths

def load_csv_files_from_list(labels_filepaths):

    list_of_df = []

    for filepath in labels_filepaths:
        df = pd.read_csv(filepath)[["filename", "relative_lx", "relative_ly", "relative_rx", "relative_ry"]]
        list_of_df.append(df)

    return list_of_df

def inspect_odd_labels(list_of_df, img_folder):

    negatives = []

    for df in list_of_df:

        negative_num = df[df["relative_lx"]<0]

        if negative_num.shape[0]>0:
            negatives.append(negative_num)

    print(len(negatives))

    for i in negatives:
        print(i.shape)
        print(i.head)

        # for row in range(i.shape[0]):
        #     filepath = i["filename"][row]
        #     im = cv2.imread(os.path.join(img_folder, filepath))
        #     cv2.imshow('im', image)
        #     cv2.waitKey(0)

    return negatives

if __name__ == '__main__':

    # initialise key filepaths
    root_folder = os.getcwd()
    img_folder = os.path.join(root_folder, "data", "processed", "mnt", "cropped_eye_imgs")
    labels_folder = os.path.join(root_folder, "data", "processed", "combined_labels")

    # return labels that there are existing images for
    labels_filepaths = get_corresponding_labels(labels_folder, img_folder)
    labels_df = load_csv_files_from_list(labels_filepaths)
    # inspect_odd_labels(labels_df, img_folder)
