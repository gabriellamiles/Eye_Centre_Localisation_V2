import os 
import tensorflow as tf
import numpy as np

import config
import model_utils
import models
from eye_data import Dataset

if __name__ == '__main__':

    # initialise key filepaths
    test_parameters = model_utils.test_configuration(os.path.join(os.getcwd(), "src", "models", "single_eye_model_training_runs.csv"), os.path.join("models", "single_eye"))
    model_input_dim = (test_parameters["input_dim"], test_parameters["input_dim"], 3)
    test_num = test_parameters["test_num"]
    batch_size=int(test_parameters["batch_size"])
    augmentation = test_parameters["augmentation"]

    save_folder = os.path.join(os.getcwd(), "models", "single_eye")

    print("*********************************")
    print(test_parameters)

    # load labels
    labels = model_utils.load_labels() # make sure label_folder is defined as correct location (in config file)
    labels = labels.sample(frac=1, random_state=42) # shuffle labels
    
    # initialise dataset
    dataset = Dataset(labels=labels)
    print(dataset.left_eye_centres.shape)

    if test_parameters["eye"]=="left":

        dataset.remove_blinks_single_eye(dataset.left_eye_centres, "left")
        dataset.get_train_val_test(dataset.single_eye_no_blinks) # split into train/test data, retrieving only single eye targets

    elif test_parameters["eye"]=="right":

        dataset.remove_blinks_single_eye(dataset.right_eye_centres, "right")
        dataset.get_train_val_test(dataset.single_eye_no_blinks)

    else:
        print("Minor problem ")

    print(dataset.single_eye_no_blinks.shape)

    dataset.get_k_folds() # apply k fold cross validation to training sets (folds = 5)

    train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[test_parameters["fold"]]], dataset.train_labels.iloc[dataset.kf_val_indices[test_parameters["fold"]]]
    print(train_labels.head())
    print(val_labels.head())
    print(train_labels.shape, val_labels.shape)
    
    # load corresponding images to train/validation sets
    train_imgs, train_labels = dataset.get_cropped_images(train_labels, test_parameters["input_dim"])
    val_imgs, val_labels = dataset.get_cropped_images(val_labels, test_parameters["input_dim"])

    # load models to train

    if test_parameters["model"] == "vgg":
        vgg = models.VGG_model(model_input_dim, test_num=test_num, output_parameters=2, batch_size=batch_size, directory=save_folder)
        vgg.train_model_memory(train_imgs, train_labels, val_imgs, val_labels)
        vgg.plot_loss_curves(directory=save_folder)

    elif test_parameters["model"] == "resnet50":
        resnet50 = models.VGG_model(model_input_dim, test_num=test_num, output_parameters=2, batch_size=batch_size, directory=save_folder)
        resnet50.train_model_memory(train_imgs, train_labels, val_imgs, val_labels)
        resnet50.plot_loss_curves(directory=save_folder)
    
    elif test_parameters["model"] == "inception":
        inception = models.VGG_model(model_input_dim, test_num=test_num, output_parameters=2, batch_size=batch_size, directory=save_folder)
        inception.train_model_memory(train_imgs, train_labels, val_imgs, val_labels)
        inception.plot_loss_curves(directory=save_folder)

    elif test_parameters["model"] == "inception_resnet_v2":
        inception_resnet_v2 = models.VGG_model(model_input_dim, test_num=test_num, output_parameters=2, batch_size=batch_size, directory=save_folder)
        inception_resnet_v2.train_model_memory(train_imgs, train_labels, val_imgs, val_labels)
        inception_resnet_v2.plot_loss_curves(directory=save_folder)

    elif test_parameters["model"] == "xception":
        xception = models.VGG_model(model_input_dim, test_num=test_num, output_parameters=2, batch_size=batch_size, directory=save_folder)
        xception.train_model_memory(train_imgs, train_labels, val_imgs, val_labels)
        xception.plot_loss_curves(directory=save_folder)