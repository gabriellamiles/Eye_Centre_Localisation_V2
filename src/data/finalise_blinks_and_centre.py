import os
import pandas as pd

if __name__=="__main__":

    folder = os.path.join(os.getcwd(), "output", "fin_eye_movement_traces_inc_blinks")
    save_folder = os.path.join(os.getcwd(), "output", "final_eye_movement_traces")

    for i in os.listdir(folder):

        print(i)
        df = pd.read_csv(os.path.join(folder, i))[["filename", "abs_lx", "abs_ly", "abs_rx", "abs_ry", "open_eyes"]]
        print(df.head())

        blinks = df.index[df["open_eyes"]==0].to_list()

        for idx in blinks:

            if idx == 0:
                continue

            last_true_val = idx - 1
            print(df.iloc[last_true_val])
            print(df[["abs_lx", "abs_ly", "abs_rx", "abs_ry"]].iloc[idx])# = df[["abs_lx", "abs_ly", "abs_rx", "abs_ry"]].iloc[last_true_val]
            print(df[["abs_lx", "abs_ly", "abs_rx", "abs_ry"]].iloc[last_true_val])

            df["abs_lx"].iloc[idx] =  df["abs_lx"].iloc[last_true_val]
            df["abs_ly"].iloc[idx] =  df["abs_ly"].iloc[last_true_val]
            df["abs_rx"].iloc[idx] =  df["abs_rx"].iloc[last_true_val]
            df["abs_ry"].iloc[idx] =  df["abs_ry"].iloc[last_true_val]
        
        df.to_csv(os.path.join(save_folder, i))