import os 
import models

import pandas as pd

def load_csv_files(directory):

    # participants = ["053", "055", "202", "204", "211"]
    list_of_df = []


    for i in os.listdir(directory):

        # for participant in participants:
        # if participant in i:

        df = pd.read_csv(os.path.join(directory, i))
        list_of_df.append(df)

    return list_of_df


if __name__ == "__main__": 

    # get labels
    labels_folder = os.path.join(os.getcwd(), "data", "raw", "final_dataset_recent")
    list_of_df = load_csv_files(labels_folder)
    print(len(list_of_df))

    # load model
    model_path = os.path.join(os.getcwd(), "models", "single_eye", "test_0_inception_20230701_143553")
    model = models.InceptionV3()