import os
import config
import pandas as pd
import utils

from sklearn.metrics import mean_squared_error

class ErrorReviewer(object):

    def __init__(self):
        
        self.full_results = self.load_results()
        self.calculate_errors()
        self.biggest_errors = self.get_biggest_errors()
        self.inspect_biggest_errors()
        # print(self.errors.shape, self.biggest_errors.shape, self.results.shape)
        # print(self.errors.head(), self.biggest_errors.head(), self.results.head())

    def calculate_errors(self):

        # check results as they stand
        y_gth = self.full_results[["orig_x", "orig_y"]]
        y_pred = self.full_results[["pred_x", "pred_y"]]
        mse = mean_squared_error(y_gth, y_pred)
        print("Mean Squared Error: " + str(mse))

        # now check after removing blinks
        self.results_no_blinks = self.full_results.copy(deep=True)
        self.results_no_blinks = self.results_no_blinks[self.results_no_blinks["orig_x"]>0]
        y_gth = self.results_no_blinks[["orig_x", "orig_y"]]
        y_pred = self.results_no_blinks[["pred_x", "pred_y"]]

        mse = mean_squared_error(y_gth, y_pred)
        print("Mean Squared Error: " + str(mse))


    def load_results(self): 

        directory = os.path.join(config.model_save_folder, config.model_name)
        
        model_data = []

        for filepath in ["full_results"]:
            _ = os.path.join(directory, filepath+".csv")
            _df = pd.read_csv(_, encoding="utf-8")
            model_data.append(_df)

        return model_data[0]

    def get_biggest_errors(self):

        # all_errors = pd.concat([self.full_results["filenames"], self.full_results[["diff_lx", "diff_ly", "diff_rx", "diff_ry"]].abs()], axis=1)

        self.full_results["diff_x"] = self.full_results["pred_x"] - self.full_results["orig_x"]
        self.full_results["diff_y"] = self.full_results["pred_y"] - self.full_results["orig_y"]

        biggest_errors = self.full_results[
            (self.full_results["diff_x"] > 10) | (self.full_results["diff_x"] < -10) |
            (self.full_results["diff_y"] > 10) | (self.full_results["diff_y"] < -10) 
        ][["filenames", "orig_x", "orig_y", "pred_x", "pred_y", "diff_x", "diff_y"]]

        # biggest_errors = biggest_errors[
        #     (self.full_results["diff_x"] > 5) | (self.full_results["diff_x"] < -5) |
        #     (self.full_results["diff_y"] > 5) | (self.full_results["diff_y"] < -5) 
        # ][["filenames", "orig_x", "orig_y", "pred_x", "pred_y", "diff_x", "diff_y"]]

        print(biggest_errors.head())
        biggest_errors.to_csv("medium_errors.csv")

        biggest_errors.columns = ["filenames", "orig_x", "orig_y", "pred_x", "pred_y", "diff_x", "diff_y"]

        print(self.full_results.shape, biggest_errors.shape)

        return biggest_errors

    def inspect_biggest_errors(self):

        retrain_list, relabel_list = [], []
        
        row=0
        for img_filepath in self.biggest_errors["filenames"]:

            print(str(row) +"/"+ str(self.biggest_errors.shape[0]))
            
            full_img_filepath = os.path.join(os.getcwd(), "data", "processed", "cropped_imgs", img_filepath)
            eye_gth = (self.biggest_errors["orig_x"].iloc[row], self.biggest_errors["orig_y"].iloc[row])
            eye_preds = (self.biggest_errors["pred_x"].iloc[row], self.biggest_errors["pred_y"].iloc[row])

            retrain = utils.review_data(full_img_filepath, eye_gth, eye_preds)


            if retrain == 1:
                print("sdave")
                retrain_list.append([img_filepath, self.biggest_errors["orig_x"].iloc[row], self.biggest_errors["orig_y"].iloc[row]])
            elif retrain == 2:
                relabel_list.append([img_filepath, self.biggest_errors["orig_x"].iloc[row], self.biggest_errors["orig_y"].iloc[row]])

            row+=1
        
        retrain_list = pd.DataFrame(retrain_list)
        retrain_list.columns = ["filename", "orig_x", "orig_y"]
        retrain_list.to_csv("retrain.csv")

        try:
            relabel_list = pd.DataFrame(relabel_list)
            relabel_list.columns = ["filename", "orig_x", "orig_y"]
            relabel_list.to_csv("relabel.csv")
        except:
            print("No images to relabel")

if __name__ == '__main__':

    error_review = ErrorReviewer()
