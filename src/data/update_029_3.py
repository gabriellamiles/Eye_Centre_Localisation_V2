import os
import pandas as pd


if __name__ == '__main__':

    file_filepath = os.path.join(os.getcwd(), "data", "raw", "final_dataset_recent", "029_3.csv")
    df = pd.read_csv(file_filepath)[["filename", "resized_LE_bottom", "resized_LE_left", "resized_LE_top", "resized_LE_right", 
                                     "resized_RE_bottom", "resized_RE_left", "resized_RE_top", "resized_RE_right", "lx", "ly", "rx", "ry", 
                                     "relative_lx", "relative_ly", "relative_rx", "relative_ry"]]

    df["filename"] = df["filename"].str.replace("bmp", "jpg")
    df.to_csv(file_filepath)

    print(df.head())