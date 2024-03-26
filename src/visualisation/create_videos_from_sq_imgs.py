"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os
import cv2

import pandas as pd
import numpy as np

from tensorflow.keras.utils import load_img

def create_video(df, img_folder, save_folder):

    if not df.shape[0] > 0:

        print(df.head)
        print(img_folder)
        return print("Empty dataframe!")
    
    video_title = (df.iloc[0, 0].split("eme1_square_imgs/")[1])
    video_title = video_title.split("/")[0] + "_" + video_title.split("/")[1] + ".avi"

    participants_already_done = ["216", "217", "218", "219", "220", "221", "222", "223", "224", "225"]
    for participant_num in participants_already_done:
        if participant_num in video_title:
            return 0

    if os.path.exists(os.path.join(save_folder, video_title)):
        return print("Video already created: " + str(video_title))
    
    out = cv2.VideoWriter(os.path.join(save_folder, video_title), cv2.VideoWriter_fourcc('M','J','P','G'), 60, (960,960))

    for row in range(df.shape[0]):
        print(str(row) + "/" + str(df.shape[0]))
        filename = str(df.iloc[row, 0])

        print(filename)

        pil_image = load_img(filename)
        open_cv_image = np.array(pil_image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

        out.write(open_cv_image)

    out.release()

if __name__ == '__main__':

    # initialise key filepaths
    model_directory = "20230208_175022"
    root_folder = os.getcwd()
    img_folder = os.path.join(root_folder, "data", "processed", "mnt", "eme1_square_imgs").replace("Eye_Centre_Localisation_V2", "Blink_Detector")
    print(".........")
    print(img_folder)
    save_folder = os.path.join(root_folder, "src", "visualisation", "eme1_square_img_videos")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # get list of img filepaths (for each participant trial)
    for participant_folder in os.listdir(img_folder):
        # for trial in ["0", "1", "2", "3"]:
        for trial in ["exp0", "exp1"]:

            trial_folder = os.path.join(img_folder, participant_folder, trial)
            if not os.path.exists(trial_folder):
                continue

            img_filepaths = sorted([os.path.join(trial_folder, i) for i in os.listdir(trial_folder)])

            first_1000_imgs_df = pd.DataFrame(data=img_filepaths[:160])

            create_video(first_1000_imgs_df, trial_folder, save_folder)
