import os
import pandas as pd


if __name__ == "__main__":

    bounding_box_folder = os.path.join(os.getcwd(), "data", "raw", "resized_bb")
    eye_centres_folder = os.path.join(os.getcwd(), "data", "raw", "combined_centres")
    save_folder = os.path.join(os.getcwd(), "data", "raw", "final_dataset_recent")

    # get bounding box and eye centres folders matching participants
    matching_participants = []

    for file in os.listdir(bounding_box_folder):

        if file in os.listdir(eye_centres_folder):
            matching_participants.append(file)

    # combine bounding boxes and absolute eye centres
    for file in matching_participants:

        bb_df = pd.read_csv(os.path.join(bounding_box_folder, file))
        centre_df = pd.read_csv(os.path.join(eye_centres_folder, file))

        print(bb_df.head())
        print(centre_df.head())

        # get dataframe of matching filenames

        fin_centres_df = pd.DataFrame()
        fin_bb_df = pd.DataFrame()

        for filename in centre_df["filename"]:

            tmp_bb_df = bb_df[bb_df['filename'].str.contains(filename)]
            tmp_centre_df = centre_df[centre_df["filename"].str.contains(filename)][["lx", "ly", "rx", "ry"]]


            fin_bb_df = pd.concat([fin_bb_df, tmp_bb_df], axis=0)
            fin_centres_df = pd.concat([fin_centres_df, tmp_centre_df], axis=0)

        fin_df = pd.concat([fin_bb_df.reset_index(drop=True), fin_centres_df.reset_index(drop=True)], axis=1)#, ignore_index=True)

        # work out relative (lx, ly, rx, ry)
        fin_df["relative_lx"] = fin_df["lx"] - fin_df["resized_LE_left"]
        fin_df["relative_ly"] = fin_df["ly"] - fin_df["resized_LE_top"]
        fin_df["relative_rx"] = fin_df["rx"] - fin_df["resized_RE_left"]
        fin_df["relative_ry"] = fin_df["ry"] - fin_df["resized_RE_top"]

        # sort out formatting for saving nicely
        fin_df = fin_df[["filename", "resized_LE_bottom", "resized_LE_left", "resized_LE_top", "resized_LE_right", "resized_RE_bottom", "resized_RE_left", "resized_RE_top", "resized_RE_right", 
                         "lx", "ly", "rx", "ry", "relative_lx", "relative_ly", "relative_rx", "relative_ry"]]
        fin_df.columns = ["filename", "resized_LE_bottom", "resized_LE_left", "resized_LE_top", "resized_LE_right", "resized_RE_bottom", "resized_RE_left", "resized_RE_top", "resized_RE_right",
                         "lx", "ly", "rx", "ry", "relative_lx", "relative_ly", "relative_rx", "relative_ry"]
        fin_df = fin_df.sort_values(by=["filename"])

        fin_df.to_csv(os.path.join(save_folder, file))