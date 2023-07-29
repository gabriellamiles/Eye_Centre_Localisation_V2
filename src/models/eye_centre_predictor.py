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
    weights_filepath = os.path.join(os.getcwd(), "models/single_eye/test_0_inception_20230702_060538/49-0.0003.hdf5")
    inception_estimator = models.Inception_model((img_dim, img_dim, 3), test_num=0, output_parameters=2, batch_size=batch_size, directory="models/single_eye")
    inception_estimator.build_model(0)
    inception_estimator.base_model_trainable()
    inception_estimator.load_trained_model(weights_filepath)
    inception_estimator.compile_model()

    # predict on unseen data
    prediction_folder_csv_files = os.path.join(os.getcwd(), "data", "raw", "all_patches")

    # labels for my computer
    csv_files_to_label = ['001_0.csv', '001_1.csv', '002_1.csv', '004_1.csv', '004_3.csv', '005_1.csv', '005_2.csv', '005_3.csv', '006_0.csv', '006_1.csv', '006_2.csv', '006_3.csv', '008_0.csv', '008_1.csv', '009_2.csv', '009_3.csv', '010_0.csv', '010_2.csv', '012_0.csv', '012_3.csv', '013_2.csv', '013_3.csv', '014_1.csv', '015_0.csv', '015_1.csv', '015_2.csv', '016_0.csv', '016_3.csv', '018_0.csv', '018_1.csv', '018_2.csv', '018_3.csv', '019_1.csv', '020_3.csv', '021_0.csv', '021_1.csv', '021_2.csv', '022_2.csv', '022_3.csv', '024_2.csv', '024_3.csv', '025_1.csv', '025_3.csv', '026_1.csv', '026_2.csv', '027_1.csv', '028_0.csv', '028_1.csv', '029_0.csv', '029_2.csv', '030_1.csv', '031_0.csv', '031_1.csv', '031_2.csv', '032_1.csv', '032_3.csv', '033_1.csv', '033_2.csv', '033_3.csv', '034_1.csv', '034_3.csv', '035_2.csv', '036_1.csv', '036_2.csv', '037_1.csv', '037_3.csv', '039_1.csv', '039_2.csv', '040_0.csv', '040_1.csv', '040_2.csv', '041_2.csv', '042_1.csv', '042_2.csv', '042_3.csv', '043_0.csv', '043_2.csv', '043_3.csv', '044_0.csv', '044_3.csv', '045_1.csv', '046_3.csv', '047_0.csv', '047_3.csv', '048_0.csv', '048_3.csv', '049_0.csv', '049_2.csv', '050_1.csv', '050_3.csv', '051_0.csv', '051_1.csv', '051_3.csv']
    
    # labels for the beast
    # csv_files_to_label = ['052_1.csv', '053_1.csv', '053_2.csv', '053_3.csv', '054_1.csv', '055_0.csv', '055_1.csv', '056_2.csv', '056_3.csv', '057_1.csv', '057_3.csv', '058_0.csv', '060_0.csv', '060_3.csv', '062_1.csv', '062_2.csv', '063_0.csv', '063_1.csv', '063_2.csv', '064_1.csv', '064_2.csv', '064_3.csv', '065_2.csv', '065_3.csv', '066_0.csv', '066_2.csv', '068_3.csv', '070_0.csv', '070_2.csv', '070_3.csv', '071_3.csv', '072_0.csv', '072_1.csv', '072_2.csv', '072_3.csv', '073_0.csv', '073_2.csv', '073_3.csv', '074_1.csv', '074_3.csv', '075_2.csv', '075_3.csv', '077_1.csv', '077_3.csv', '200_1.csv', '200_2.csv', '200_3.csv', '202_0.csv', '202_1.csv', '202_3.csv', '203_3.csv', '204_0.csv', '205_0.csv', '205_3.csv', '206_3.csv', '207_0.csv', '207_1.csv', '208_0.csv', '208_2.csv', '208_3.csv', '209_0.csv', '209_2.csv', '209_3.csv', '210_0.csv', '210_2.csv', '211_1.csv', '211_2.csv', '211_3.csv', '212_0.csv', '212_1.csv', '212_2.csv', '213_0.csv', '213_2.csv', '214_0.csv', '214_3.csv', '216_1.csv', '216_3.csv', '217_2.csv', '218_0.csv', '218_1.csv', '221_2.csv', '222_0.csv', '222_1.csv', '222_3.csv', '225_0.csv', '225_1.csv', '225_3.csv', '226_0.csv', '226_2.csv', '226_3.csv', '227_1.csv', '227_2.csv', '229_3.csv']

    csv_filepaths = [os.path.join(prediction_folder_csv_files, i) for i in csv_files_to_label]

    # print(csv_filepaths)
    # csv_filepaths = [os.path.join(prediction_folder_csv_files, i) for i in os.listdir(prediction_folder_csv_files)]
    img_folder = os.path.join(os.getcwd().replace("Eye_Centre_Localisation_V2", "Eye_Region_Detection"), "data", "processed", "mnt", "eye_patches")

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

        if count == 5:
            break


