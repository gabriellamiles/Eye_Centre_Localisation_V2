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
    dataset.get_train_val_test() # split into train/test data
    dataset.get_k_folds() # apply k fold cross validation to training sets (folds = 5)

    # for each fold, load images and train models
    for i in range(len(dataset.kf_train_indices)):
        print(f"Fold: {i}")
        train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[i]], dataset.train_labels.iloc[dataset.kf_val_indices[i]]
        
        train_filenames, train_labels = train_labels[["filename"]], train_labels[config.eye_centre_cols]/960
        val_filenames, val_labels = val_labels[["filename"]], val_labels[config.eye_centre_cols]/960

        # load corresponding images at size (224, 224, 3)
        dataset.get_train_val_images(train_filenames=train_filenames, val_filenames=val_filenames)
        train_imgs = (np.asarray(dataset.train_imgs)).astype(np.float32)
        val_imgs = (np.asarray(dataset.val_imgs)).astype(np.float32)

        # train models with default size of (224, 224, 3)
        # vgg model
        tf.keras.backend.clear_session()
        # del vgg
        vgg = models.VGG_model((224, 224, 3), output_parameters=4)
        vgg.train_model(train_imgs, train_labels, val_imgs, val_labels)
        vgg.plot_loss_curves()
        

        # resnet model
        # resnet_50 = models.ResNet50_model((224, 224, 3), output_parameters=4)
        # resnet_50.train_model(train_imgs, train_labels, val_imgs, val_labels)
        # resnet_50.plot_loss_curves()


    # for each fold, load images and train models
    for i in range(len(dataset.kf_train_indices)):
        print(f"Fold: {i}")
        train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[i]], dataset.train_labels.iloc[dataset.kf_val_indices[i]]
        
        train_filenames, train_labels = train_labels[["filename"]], train_labels[config.eye_centre_cols]/960
        val_filenames, val_labels = val_labels[["filename"]], val_labels[config.eye_centre_cols]/960

        # load corresponding images at size (299, 299, 3)
        dataset.get_train_val_images(train_filenames=train_filenames, val_filenames=val_filenames, target_size=(299, 299, 3))
        train_imgs = np.asarray(dataset.train_imgs).astype(np.float32)
        val_imgs = np.asarray(dataset.val_imgs).astype(np.float32)

        # train models with default size of (299, 299, 3)
        # inception model
        inception = models.Inception_model((299, 299, 3), output_parameters=4)
        inception.train_model(train_imgs, train_labels, val_imgs, val_labels)
        inception.plot_loss_curves()

        # inception with residual connections
        inception_resnet_v2 = models.InceptionResNetV2((299, 299, 3), output_parameters=4)
        inception_resnet_v2.train_model(train_imgs, train_labels, val_imgs, val_labels)
        inception_resnet_v2.plot_loss_curves()

        # xception model
        xception = models.Xception_model((299, 299, 3))
        xception.train_model(train_imgs, train_labels, val_imgs, val_labels)
        xception.plot_loss_curves()

        