"Gabriella Miles, Farscope Phd, Bristol Robotics Laboratory"

import os
import numpy as np
import pandas as pd
import cv2

# initialise key filepaths
root_folder = os.getcwd()
label_folder = os.path.join(root_folder, "data", "processed", "combined_labels")
label_filepaths = [os.path.join(label_folder, i) for i in os.listdir(label_folder)]
img_folder = os.path.join(root_folder, "data", "processed", "mnt", "eme2_square_imgs")

files_to_amend = []
for filepath in label_filepaths:
    print(filepath)
    df = pd.read_csv(filepath)

    if os.path.exists(filepath.replace("combined_labels", "checked_labels")): 
        print("Already reviewed:" + str(filepath) +".")
        continue
    
    correctness= []
    for row in range(df.shape[0]):

        filename = df["filename"].iloc[row]
        lx = df["lx"].iloc[row]
        ly = df["ly"].iloc[row]
        rx = df["rx"].iloc[row]
        ry = df["ry"].iloc[row]

        im = cv2.imread(os.path.join(img_folder, filename))

        cv2.circle(im, (int(lx), int(ly)), 4, (255, 0, 0), -1)
        cv2.circle(im, (int(rx), int(ry)), 4, (0, 255, 0), -1)

        cv2.imshow("image", im)

        a = cv2.waitKey(0)

        print(filename)

        if a == ord("y"):
            print("Centre coordinates correct")
            correctness.append("Y")
            cv2.destroyAllWindows()
        elif a == ord("n"):
            print("Centre coordinates wrong.")
            correctness.append("N")
            cv2.destroyAllWindows()
        elif a == ord("s"):
            print("Skipping participant.")
            break

    df["correctness"] = correctness
    df.to_csv(filepath.replace("combined_labels", "checked_labels"))
