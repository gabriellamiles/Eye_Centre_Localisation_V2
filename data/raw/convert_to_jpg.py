import os
import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv(os.path.join(os.getcwd(), "029_3.csv"))


    count = 0
    for row in df["filename"]:

        print(row)
        new_row = row.replace("bmp", "jpg")
        print(new_row)

        df["filename"].iloc[count] = new_row

        count +=1

    df[["filename", "relative_LE_left", "relative_LE_top", "relative_LE_right",	"relative_LE_bottom", "relative_RE_left", "relative_RE_top", "relative_RE_right", "relative_RE_bottom",	"lx", "ly", "rx", "ry",	"relative_lx", "relative_ly", "relative_rx", "relative_ry"]].to_csv("new029_3.csv")
