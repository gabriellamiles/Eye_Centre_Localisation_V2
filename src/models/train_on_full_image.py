import os 
import tensorflow as tf
import numpy as np
import pandas as pd

import config
import model_utils
import models
from eye_data import Dataset

if __name__ == '__main__':

    test_parameters = model_utils.test_configuration(os.path.join(os.getcwd(), "src", "models", "model_training_runs.csv"))

    # load labels
    labels = model_utils.load_labels() # make sure label_folder is defined as correct location (in config file)
    labels = labels.sample(frac=1, random_state=42) # shuffle labels
    
    # initialise dataset
    dataset = Dataset(labels=labels)
    dataset.get_train_val_test() # split into train/test data
    dataset.get_k_folds() # apply k fold cross validation to training sets (folds = 5)

    # load relevant fold
    train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[test_parameters["fold"]]], dataset.train_labels.iloc[dataset.kf_val_indices[test_parameters["fold"]]]
    train_filenames, train_labels = train_labels[["filename"]], train_labels[config.eye_centre_cols]/960
    val_filenames, val_labels = val_labels[["filename"]], val_labels[config.eye_centre_cols]/960

    # load corresponding images at correct size for model
    dataset.get_train_val_images(train_filenames=train_filenames, val_filenames=val_filenames, target_size=(test_parameters["input_dim"], test_parameters["input_dim"], 3))
    train_imgs = (np.asarray(dataset.train_imgs)).astype(np.float32)
    val_imgs = (np.asarray(dataset.val_imgs)).astype(np.float32)

    print("**********************************")
    print(test_parameters)
    print("**********************************")

    if test_parameters["model"] == "vgg":
        # load model to train
        vgg = models.VGG_model((test_parameters["input_dim"], test_parameters["input_dim"], 3), test_num=test_parameters["test_num"], output_parameters=4, batch_size=int(test_parameters["batch_size"]))
        vgg.train_model(train_imgs, train_labels, val_imgs, val_labels)
        vgg.plot_loss_curves()

    elif test_parameters["model"] == "resnet50":    
        # resnet model
        resnet_50 = models.ResNet50_model((test_parameters["input_dim"], test_parameters["input_dim"], 3), test_num=test_parameters["test_num"], output_parameters=4, batch_size=int(test_parameters["batch_size"]))
        resnet_50.train_model(train_imgs, train_labels, val_imgs, val_labels)
        resnet_50.plot_loss_curves()

    elif test_parameters["model"] == "inception":
        # inception model
        inception = models.Inception_model((test_parameters["input_dim"], test_parameters["input_dim"], 3), test_num=test_parameters["test_num"], output_parameters=4, batch_size=int(test_parameters["batch_size"]))
        inception.train_model(train_imgs, train_labels, val_imgs, val_labels)
        inception.plot_loss_curves()

    elif test_parameters["model"] == "inception_resnet_v2":
        # inception with residual connections
        inception_resnet_v2 = models.InceptionResNetV2_model((test_parameters["input_dim"], test_parameters["input_dim"], 3), test_num=test_parameters["test_num"], output_parameters=4, batch_size=int(test_parameters["batch_size"]))
        inception_resnet_v2.train_model(train_imgs, train_labels, val_imgs, val_labels)
        inception_resnet_v2.plot_loss_curves()

    elif test_parameters["model"] == "xception":
        # xception model
        xception = models.Xception_model((test_parameters["input_dim"], test_parameters["input_dim"],  3), test_num=test_parameters["test_num"], output_parameters=4, batch_size=int(test_parameters["batch_size"]))
        xception.train_model(train_imgs, train_labels, val_imgs, val_labels)
        xception.plot_loss_curves()

        