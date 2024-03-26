import config
import model_utils
import models
import os
import numpy as np
import tensorflow as tf
import pandas as pd
# import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from eye_data import Dataset
from data_generator import CustomDataGenerator, SingleEyeDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier
from pathlib import Path

def load_dataset():

    # load labels
    labels = model_utils.load_labels() # make sure label_folder is defined as correct location (in config file)
    labels = labels.sample(frac=1, random_state=42) # shuffle labels

    # initialise dataset
    dataset = Dataset(labels=labels)

    # initial preprocessing here
    # dataset.remove_blinks()
    dataset.get_train_val_test(dataset.eye_centres) # split into train/test data
    dataset.get_k_folds() # apply k fold cross validation to training sets (folds = 5)

    return dataset

def load_data_generators(img_dim, train_filenames, train_labels, val_filenames, val_labels, test_filepath, val_batch_size=1):
    # initialise data generators

    eye_data_generator_train = SingleEyeDataGenerator(
            batch_size=batch_size, 
            x_set=train_filenames, 
            y_set=train_labels, 
            root_directory=test_filepath,
            img_dim=img_dim, 
            augmentation="nothing",
            shuffle=True
            )
    
    eye_data_generator_val = SingleEyeDataGenerator(
            batch_size=val_batch_size, 
            x_set=val_filenames, 
            y_set=val_labels, 
            root_directory=test_filepath,
            img_dim=img_dim, 
            augmentation="nothing",
            shuffle=False
            )

    
    return eye_data_generator_train, eye_data_generator_val

def get_test_data_generator(test_filenames):
    eye_data_generator_test = SingleEyeDataGenerator(
            batch_size=1, 
            x_set=test_filenames, 
            root_directory=test_filepath,
            img_dim=img_dim, 
            augmentation="nothing",
            shuffle=False
            )

def examine_batches(generator, dim):

    images = next(iter(generator))
    nrows, ncols = 4, 2
    
    fig = plt.figure(figsize=(10,10))

    for i in range(8):

        ax = fig.add_subplot(nrows, ncols, i+1)
        plt.imshow(images[0][i].astype('uint8'))
        print(images[1][i])
        circ1 = Circle((images[1][i,0]*dim, images[1][i,1]*dim),5)
        ax.add_patch(circ1)

        plt.axis(False)

    plt.savefig("test.png")
    plt.show()

def get_fold_data(dataset, k_fold):

    # get relevant training and validation folds  
    train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[k_fold]], dataset.train_labels.iloc[dataset.kf_val_indices[k_fold]]
    train_filenames, train_labels = train_labels[["filename"]], train_labels[["x", "y"]]
    val_filenames, val_labels = val_labels[["filename"]], val_labels[["x", "y"]]

    return train_filenames, train_labels, val_filenames, val_labels

# def inspect_single_eye_images(train_set_filepath, img_folder):

#     df = pd.read_csv(train_set_filepath)[["filename", "x", "y"]]

#     for row in range(0, df.shape[0]):

#         img_filepath = df["filename"].iloc[row]
#         full_img_filepath = os.path.join(img_folder, img_filepath)

#         x = df["x"].iloc[row]
#         y = df["y"].iloc[row]

#         im = cv2.imread(full_img_filepath)
#         cv2.circle(im, (x,y), 4, (255, 0, 0), -1)

#         cv2.imshow("im", im)


#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


if __name__ == '__main__':

    # initialise key filepaths and hyperparameters/key parameters
    test_filepath = os.path.join(os.getcwd(), "data", "processed", "cropped_imgs")
    batch_size = 64
    train_set_filepath = os.path.join(os.getcwd(), "data", "processed", "train_split.csv")
    img_folder = os.path.join(os.getcwd(), "data", "processed", "cropped_imgs")

    # inspect_single_eye_images(train_set_filepath, img_folder)

    # # load dataset 
    # dataset = load_dataset()
    # print(dataset.single_eye_df.head())
    # dataset.get_train_val_test(dataset.single_eye_df[["filename", "x", "y"]])

    # # print(dataset.val_labels)
    # # dataset.get_k_folds(dataset.train_labels)

    # # remove blinks
    # print("Dataset size: ")
    # print(dataset.train_labels.shape, dataset.val_labels.shape)
    # dataset.full_train_labels = dataset.full_train_labels[dataset.full_train_labels["x"]>0]
    # # dataset.val_labels = dataset.val_labels[dataset.val_labels["x"]>0]
    # dataset.test_labels = dataset.test_labels[dataset.test_labels["x"]>0]
    # print(dataset.train_labels.shape, dataset.val_labels.shape)

    # # train and predict with model
    # train_filenames, train_labels, = dataset.full_train_labels[["filename"]], dataset.full_train_labels[["x", "y"]]
    # # val_filenames, val_labels, = dataset.val_labels[["filename"]], dataset.val_labels[["x", "y"]]
    # test_filenames, test_labels, = dataset.test_labels[["filename"]], dataset.test_labels[["x", "y"]]

    

    # # load data generators
    img_dim = 299
    # eye_data_generator_train, eye_data_generator_val = load_data_generators(img_dim, train_filenames, train_labels, test_filenames, test_labels, test_filepath, val_batch_size=1)
    
    # # # examine output of generators and labels
    # # examine_batches(eye_data_generator_train, 299)

    # for test in range(0, 1):

    #     # construct model
    #     inception_estimator = models.Inception_model((img_dim, img_dim, 3), test_num=test, output_parameters=2, batch_size=batch_size, directory="models/single_eye")
    #     # build and compile
    #     inception_estimator.build_model(test)
    #     inception_estimator.compile_model()
        
    #     # train model
    #     inception_estimator.train_model(eye_data_generator_train, eye_data_generator_val)
    #     # inception_estimator.predict_model(eye_data_generator_val)
    #     # inception_estimator.plot_loss_curves(directory="models/single_eye")
    #     # inception_estimator.save_results(val_labels, val_filenames)

    #     inception_estimator.base_model_trainable()
    #     inception_estimator.train_model(eye_data_generator_train, eye_data_generator_val)
        
    #     # get results from model
    #     inception_estimator.predict_model(eye_data_generator_val)
    #     inception_estimator.plot_loss_curves(directory="models/single_eye")
    #     inception_estimator.save_results(test_labels, test_filenames)

    # load desired model
    weights_filepath = os.path.join(os.getcwd(), "models/single_eye_original/test_0_inception_20230701_143553/39-0.0001.hdf5")
    inception_estimator = models.Inception_model((img_dim, img_dim, 3), test_num=0, output_parameters=2, batch_size=batch_size, directory="models/single_eye")
    inception_estimator.build_model(0)
    inception_estimator.base_model_trainable()
    inception_estimator.load_trained_model(weights_filepath)
    inception_estimator.compile_model()

    # predict on unseen data
    prediction_folder_csv_files = os.path.join(os.getcwd(), "data", "raw", "right_eye_predictions")

    csv_filepaths = [os.path.join(prediction_folder_csv_files, i) for i in os.listdir(prediction_folder_csv_files)]
    img_folder = os.path.join(os.getcwd(), "data", "processed", "mnt1", "eye_patches")

    predictions_still_to_make = []
    count = 0
    for filepath in csv_filepaths:

        print(filepath)

        # check if predictions have already been made 
        check_filepath = os.path.join(os.getcwd(), "output", "patch_predictions", os.path.basename(filepath))

        if os.path.exists(check_filepath):
            print("Predictions already made for " + os.path.basename(filepath) + ". Skipping predictions.")
            continue
        else:
            predictions_still_to_make.append(os.path.basename(check_filepath))

        excluded_particiants = ["034", "038", "059", "067", "076", "077", "216", "219", "220"]
        for participant in excluded_particiants:
            if participant in filepath: 
                print("Participant excluded: " + str(filepath))
                continue
        
        df = pd.read_csv(filepath)

        pred_list = []

        for row in df["filename"]:

            print(os.path.join(img_folder, row))
            try:
                image = tf.keras.utils.load_img(os.path.join(img_folder, row))
                input_arr = tf.keras.utils.img_to_array(image)
                input_arr = np.array([input_arr])  # Convert single image to a batch.
                predictions = inception_estimator.model.predict(input_arr)*299

                print(predictions)
                print(predictions[0][0])

                pred_list.append([row, predictions[0][0], predictions[0][1]])
            except:
                continue

        try:
            pred_df = pd.DataFrame(data=pred_list, columns=["filename", "pred_x", "pred_y"])
            csv_name = filepath.replace("data/raw/all_patches", "output/patch_predictions")
            pred_df.to_csv(csv_name)
            print("Saved at: " + str(csv_name))
        except:
            continue

        count += 1

    #     if count == 5:
    #         break


