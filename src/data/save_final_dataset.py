import os
import pandas as pd
import cv2

def save_images_to_folder(df, img_folder, save_folder):

    print(df.head())

    for filepath in df["filename"]:
        
        img_path = os.path.join(img_folder, filepath)
        participant_trial_folder = os.path.dirname(img_path).split("eme2_square_imgs/")[1]
        if os.path.exists(img_path):
            print("Found")
            pass
        else:
            print("Not found")
            continue

        full_save_folder = os.path.join(save_folder, participant_trial_folder)
        # create necessary folders
        if not os.path.exists(full_save_folder):
            os.makedirs(full_save_folder)

        # save image to specified location
        im = cv2.imread(img_path)
        save_im_filepath = os.path.join(full_save_folder, os.path.basename(img_path))
        cv2.imwrite(save_im_filepath, im)


    cv2.destroyAllWindows()

if __name__ == '__main__':

    # key information
    root_folder = os.getcwd()
    eye_centre_data_folder = os.path.join(root_folder, "data", "processed", "combined_labels")
    img_folder = os.path.join(os.getcwd(), "data", "processed", "mnt", "eme2_square_imgs")
    save_folder = os.path.join(os.getcwd(), "data", "final_dataset", "images")

    print(img_folder)

    all_label_df = pd.DataFrame()

    for label_file in os.listdir(eye_centre_data_folder):

        if "013" in label_file:

            label_filepath = os.path.join(eye_centre_data_folder, label_file)
            label_df = pd.read_csv(label_filepath)[["filename", "lx", "ly", "rx", "ry"]]
            label_df = label_df[label_df["ly"]> 50]


            all_label_df = pd.concat([all_label_df, label_df], axis=0)
        else:
            print("Not participant 004 or 013.")
            print(label_file)

    print(all_label_df.shape)


    save_images_to_folder(all_label_df, img_folder, save_folder)