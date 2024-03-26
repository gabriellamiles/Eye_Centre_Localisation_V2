import os 
import pandas as pd

if __name__=="__main__":

    data_folder = os.path.join(os.getcwd(), "output", "final_eye_movement_traces")
    intermediate_folder = os.path.join(os.getcwd(), "output", "eye_movement_traces_test")

    contains_nan = []
    for i in os.listdir(intermediate_folder):

        df = pd.read_csv(os.path.join(intermediate_folder, i))[["filename", "abs_lx", "abs_ly", "abs_rx", "abs_ry", "open_eyes"]]

        check_nan = df.isnull().values.any()

        if check_nan:
            print(i)
            contains_nan.append(i)

            df1 = df[df.isna().any(axis=1)]
            # print(df1.head())
            print(df1.shape)

            if df1.shape[0] < 500: 

                df = df.ffill()
            #     df1 = df[df.isna().any(axis=1)]
                df.to_csv(os.path.join(data_folder, i))
        else:
            df.to_csv(os.path.join(data_folder, i))

    print(contains_nan)
    print(len(contains_nan))