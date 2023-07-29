import os
import cv2
import pandas as pd

import dataset_utils

def edit_imgs(fin_df, img_folder, save_folder):

    for i in range(fin_df.shape[0]):

        # load images
        filename = fin_df["filename"].iloc[i]
        # print(filename)
        full_filepath = os.path.join(img_folder, filename)
        # print(full_filepath)

        # crop images
        im = cv2.imread(full_filepath)

        # cv2.imshow("im", im)
        # cv2.waitKey(0)

        le_left = fin_df["relative_LE_left"].iloc[i]
        le_right = fin_df["relative_LE_right"].iloc[i]
        le_bottom = fin_df["relative_LE_bottom"].iloc[i]
        le_top = fin_df["relative_LE_top"].iloc[i]
    
        re_left = fin_df["relative_RE_left"].iloc[i]
        re_right = fin_df["relative_RE_right"].iloc[i]
        re_bottom = fin_df["relative_RE_bottom"].iloc[i]
        re_top = fin_df["relative_RE_top"].iloc[i]

        left_crop = im[le_top:le_bottom, le_left:le_right]

        # cv2.imshow("im", left_crop)
        # cv2.waitKey(0)

        right_crop = im[re_top:re_bottom, re_left:re_right]

        # cv2.imshow("im", right_crop)
        # cv2.waitKey(0)

        # save images in good place
        save_left_eye = os.path.join(save_folder, filename.replace(".jpg", "_left.jpg"))
        save_right_eye = os.path.join(save_folder, filename.replace(".jpg", "_right.jpg"))
        
        check_folders = save_left_eye.rsplit("/", 1)[0]
        try:
            os.makedirs(check_folders)
        except FileExistsError:
            pass

        check_folders = save_right_eye.rsplit("/", 1)[0]
        try:
            os.makedirs(check_folders)
        except FileExistsError:
            pass

        try:
            cv2.imwrite(save_left_eye, left_crop)
        except cv2.error: 
            pass

        try:
            cv2.imwrite(save_right_eye, right_crop)
        except cv2.error:
            pass

        cv2.destroyAllWindows()

        # cv2.waitKey(0)





if __name__ == '__main__':

    # initialise key filepaths
    root_folder = os.getcwd()
    save_folder = os.path.join(root_folder, "cropped_imgs")
    labels_folder = os.path.join(root_folder, "data", "raw", "final_dataset")
    img_folder = os.path.join(root_folder, "data", "processed", "mnt", "eme2_square_imgs")
    img_folder = img_folder.replace("Eye_Centre_Localisation_V2", "Eye_Region_Detection")

    # get data
    all_label_filepaths = dataset_utils.retrieve_csv_filepaths_from_directory(labels_folder)
    fin_df = dataset_utils.load_and_concatenate_list_of_df(all_label_filepaths)

    # get and save images 
    edit_imgs(fin_df, img_folder, save_folder)