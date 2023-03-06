"Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory"

import os
import cv2
import pandas as pd

if __name__ == '__main__':

    # initialise key filepaths and create folders where necessary
    root_folder = os.getcwd() # current working directory
    predictions_filepath = os.path.join(root_folder, "models", "20230210_173438", "test_set_predictions.csv")# filepath for filenames, predictions
    imgs_directory = os.path.join(root_folder, "data", "processed", "test_data", "imgs") # filepath for images
    save_directory = os.path.join(root_folder, "src", "visualisation", "predictions")# filepath to save images under

    # create save directory if required
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # load test filenames and predictions
    pred_df = pd.read_csv(predictions_filepath)[["filename", "x", "y"]]

    # load corresponding images
    for i in range(pred_df.shape[0]):
        image_filename = pred_df["filename"][i] # retrieve image filename
        x = int(pred_df["x"][i]) # predicted x coordinate
        y = int(pred_df["y"][i]) # predicted y cooordinate

        img_filepath = os.path.join(imgs_directory, image_filename) # get full filepath from working directory to image
        image = cv2.imread(img_filepath) # load image


        # edit images to include prediction: plot prediction on image
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

        # save images with overlaid prediction
        # save_filename = image_filename.rsplit("/",1)[-1]
        save_filename = str(i)+".jpg"
        save_filepath = os.path.join(save_directory, save_filename) # full filepath for image
        cv2.imwrite(save_filepath, image) # save image at specified filepath
