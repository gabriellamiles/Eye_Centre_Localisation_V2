import os 

import pandas as pd


if __name__ == '__main__':

    # initialise key filepaths
    root_folder = os.getcwd()
    relabelled_filepath = os.path.join(root_folder, "data", "raw", "relabelled.csv")
    data_folder = os.path.join(root_folder, "data", "raw", "final_dataset")
    save_folder = os.path.join(root_folder, "data", "raw", "relabelled_dataset")

    # load data
    df = pd.read_csv(relabelled_filepath)
    print(df.head())
    print(df.shape)

    # extract participant and trial number

    count = 0
    for row in df["filename"]:

        if row[-4:] == ".bmp":
            count +=1
            continue
        
        # get key info from filename
        participant = row.split("/")[0]
        trial = row.split("/")[1]

        left_eye = True
        if "right" in row: 
            left_eye=False

        img_number = (row.split("-")[-1]).split("_")[0]

        # find corresponding row in original dataframe
        corresponding_df = pd.read_csv(os.path.join(data_folder, participant+"_"+trial+".csv"))
        t = corresponding_df.index[corresponding_df["filename"].str.contains(img_number)]

        print(corresponding_df.iloc[t])

        # get new x, y values
        x = df["x"].iloc[count]
        y = df["y"].iloc[count]

        if left_eye:
            corresponding_df["relative_lx"].iloc[t] = x
            corresponding_df["relative_ly"].iloc[t] = y
        else:
            corresponding_df["relative_rx"].iloc[t] = x
            corresponding_df["relative_ry"].iloc[t] = y

        print(corresponding_df.iloc[t])

        # save updated file
        corresponding_df[["filename", "relative_LE_left", "relative_LE_top", "relative_LE_right", "relative_LE_bottom", "relative_RE_left", "relative_RE_top", "relative_RE_right", "relative_RE_bottom", "lx", "ly", "rx", "ry", "relative_lx", "relative_ly", "relative_rx", "relative_ry"]].to_csv(os.path.join(data_folder, participant+"_"+trial+".csv"))


        count+=1
        

