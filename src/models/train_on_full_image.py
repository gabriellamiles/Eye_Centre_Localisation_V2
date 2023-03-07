import os 
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from PIL import Image
from keras.utils import load_img, img_to_array

import config
import model_utils
import models


class Dataset():
    def __init__(self,
                 labels=None
                 ):
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None

        self.get_eye_centres()
        
    def get_eye_centres(self):
        """ Returns absolute eye centres."""
        self.eye_centres = self.labels[["filename", "lx", "ly", "rx", "ry"]]

    def get_train_val_test(self):
        """ Function splits labels into train/validation/test labels. """
        # train/test/validation split -- 0.6/0.2/0.2
        self.train_labels, self.test_labels = train_test_split(self.labels, test_size=0.2) # split into test and train
        # train_labels, val_labels = train_test_split(train_labels, test_size=0.2) # split into train and val
    
        # train_filenames, val_filenames = train_labels[config.filename], val_labels[config.filename]
        # train_labels, val_labels = train_labels[config.eye_centre_cols], val_labels[config.eye_centre_cols]

        # self.train_labels = train_labels/960.0 # get ratio of position in original image
        # val_labels = val_labels/960.0

        # add code to save test data set
        self.save_test_data()

    
    def get_k_folds(self):
        """Split labels into specific folds of data as relevant"""
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

        self.kf_train_indices, self.kf_val_indices = [], []
    
        for i, (train_index, val_index) in enumerate(self.kf.split(self.train_labels)):
        
            self.kf_train_indices.append(train_index)
            self.kf_val_indices.append(val_index)
    
    def get_train_val_images(self, train_filenames, val_filenames):

        self.train_imgs, self.val_imgs = [], [] # initiate empty lists
        
        count = 0
        for set_of_filepaths in [train_filenames, val_filenames]:

            for row in range(set_of_filepaths.shape[0]):

                filepath = str(set_of_filepaths.iloc[row, 0])
            

                img_path = os.path.join(config.square_img_folder, filepath)
                im = load_img(img_path, color_mode="rgb", target_size=(224, 224))
                input_arr = img_to_array(im)/255.0


                if count == 0:
                    self.train_imgs.append(input_arr)

                elif count == 1:
                    self.val_imgs.append(input_arr)

            count += 1

    def save_test_data(self):
        """ Saves csv file containing test split (filenames + targets)."""        
        self.test_labels.to_csv(config.test_split_save_location)

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
        train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[i]], dataset.train_labels.iloc[dataset.kf_val_indices[i]]
        
        train_filenames, train_labels = train_labels[["filename"]], train_labels[config.eye_centre_cols]/960
        val_filenames, val_labels = val_labels[["filename"]], val_labels[config.eye_centre_cols]/960

        # load corresponding images to train/validation sets
        dataset.get_train_val_images(train_filenames=train_filenames, val_filenames=val_filenames)
        train_imgs = (np.asarray(dataset.train_imgs)/255).astype(np.float32)
        val_imgs = (np.asarray(dataset.val_imgs)/255).astype(np.float32)
        
        print(train_labels.shape, val_labels.shape, train_filenames.shape, val_filenames.shape, train_imgs.shape, val_imgs.shape)

        # load models to train
        vgg = models.VGG_model((224, 224, 3))
        vgg.train_model(train_imgs, train_labels, val_imgs, val_labels)

    
