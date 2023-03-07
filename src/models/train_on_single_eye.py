import os 
import tensorflow as tf
import numpy as np

import config
import model_utils
import models
from eye_data import Dataset

if __name__ == '__main__':

    # load labels
    labels = model_utils.load_labels() # make sure label_folder is defined as correct location (in config file)
    labels = labels.sample(frac=1) # shuffle labels
    
    # initialise dataset
    dataset = Dataset(labels=labels)
    dataset.get_train_val_test(dataset.left_eye_centres) # split into train/test data, retrieving only single eye targets
    dataset.get_k_folds() # apply k fold cross validation to training sets (folds = 5)

    # for each fold, load images and train models
    for i in range(len(dataset.kf_train_indices)):

        train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[i]], dataset.train_labels.iloc[dataset.kf_val_indices[i]]
        print(train_labels.head())
        print(val_labels.head())
        print(train_labels.shape, val_labels.shape)
        
        # train_filenames, train_labels = train_labels[["filename"]], train_labels[config.eye_centre_cols]
        # val_filenames, val_labels = val_labels[["filename"]], val_labels[config.eye_centre_cols]

        # load corresponding images to train/validation sets
        train_imgs, train_labels = dataset.get_cropped_images(train_labels)
        val_imgs, val_labels = dataset.get_cropped_images(val_labels)

        # train_imgs = (np.asarray(dataset.train_imgs)/255).astype(np.float32)
        # val_imgs = (np.asarray(dataset.val_imgs)/255).astype(np.float32)
        
        # print(train_labels.shape, val_labels.shape, train_filenames.shape, val_filenames.shape, train_imgs.shape, val_imgs.shape)

        # # load models to train
        # vgg = models.VGG_model((224, 224, 3), output_parameters=2)
        # vgg.train_model(train_imgs, train_labels, val_imgs, val_labels)

        break