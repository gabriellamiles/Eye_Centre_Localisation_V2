import os
import pandas as pd

def retrieve_csv_filepaths_from_directory(directory):

    csv_filepaths = []

    for i in os.listdir(directory):
        if i[-4:]==".csv":
            tmp = os.path.join(directory, i)
            csv_filepaths.append(tmp)

    return csv_filepaths

def load_and_concatenate_list_of_df(list_of_df):

    fin_df = pd.DataFrame()

    participants = ["004_2"]

    for filepath in list_of_df:
        print(filepath)
        for participant in participants:
            if participant in filepath:
                df = pd.read_csv(filepath)
                print(df.shape)
                fin_df = pd.concat([fin_df, df], axis=0)

    return fin_df