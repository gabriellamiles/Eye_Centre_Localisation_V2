"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os
import pandas as pd
import numpy as np
import cv2

root_directory = os.path.join(os.getcwd(), "models")
test_folders = [os.path.join(root_directory, i) for i in os.listdir(root_directory) if "test_0" in i]
root_image_folder = os.path.join(os.getcwd(), "data", "processed", "mnt", "eme2_square_imgs")

for folder in test_folders:
    results_df = pd.read_csv(os.path.join(folder, "errors.csv"))

    results_df["acc_lx"] = results_df["lx"] - results_df["pred_lx"]
    results_df["acc_ly"] = results_df["ly"] - results_df["pred_ly"]
    results_df["acc_rx"] = results_df["rx"] - results_df["pred_rx"]
    results_df["acc_ry"] = results_df["ry"] - results_df["pred_ry"]

    # retrieve all where the error in accuracy is greater than 3 pixels
    accuracy_threshold = 3
    print(results_df.shape)
    error_df = results_df.loc[(results_df["acc_lx"]>accuracy_threshold) | (results_df["acc_ly"]>accuracy_threshold) | (results_df["acc_rx"]>accuracy_threshold) | (results_df["acc_ry"]>accuracy_threshold)]

    results_df["correctness"] = np.nan

    print(results_df.columns)

    for row in range(error_df.shape[0]):

        # extract key information
        filename = error_df["filename"].iloc[row]
        lx = error_df["lx"].iloc[row]
        ly = error_df["ly"].iloc[row]
        rx = error_df["rx"].iloc[row]
        ry = error_df["ry"].iloc[row]

        img_filepath = os.path.join(root_image_folder, filename)
        im = cv2.imread(img_filepath)

        cv2.circle(im, (int(lx), int(ly)), 4, (255, 0, 0), -1)
        cv2.circle(im, (int(rx), int(ry)), 4, (0, 255, 0), -1)

        cv2.imshow("image", im)

        a = cv2.waitKey(0)

        print(filename)

        if a == ord("y"):
            print("Centre coordinates correct")
            results_df["correctness"].iloc[row] = "Y"
            cv2.destroyAllWindows()
        elif a == ord("n"):
            print("Centre coordinates wrong.")
            results_df["correctness"].iloc[row] = "N"
            cv2.destroyAllWindows()

    # handle correctness
    print(results_df.head())

    results_df.to_csv(os.path.join(folder, "errors_with_correctness.csv"))

    break
