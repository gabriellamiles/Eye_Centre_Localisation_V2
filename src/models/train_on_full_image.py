import os 
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
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
        train_labels, self.test_labels = train_test_split(self.labels, test_size=0.2) # split into test and train
        train_labels, val_labels = train_test_split(train_labels, test_size=0.2) # split into train and val

        train_filenames, val_filenames = train_labels[config.filename], val_labels[config.filename]
        train_labels, val_labels = train_labels[config.eye_centre_cols], val_labels[config.eye_centre_cols]

        train_labels = train_labels/960.0 # get ratio of position in original image
        val_labels = val_labels/960.0

        # add code to save test data set
        self.save_test_data()

        return train_labels, val_labels, train_filenames, val_filenames
    
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
    labels = labels.sample(frac=1)
    # shuffle labels
    
    # initialise dataset
    dataset = Dataset(labels=labels)
    train_y, val_y, train_filenames, val_filenames = dataset.get_train_val_test()
    
    # load corresponding images to train/validation sets
    dataset.get_train_val_images(train_filenames=train_filenames, val_filenames=val_filenames)
    train_imgs = np.asarray(dataset.train_imgs).astype(np.float32)
    val_imgs = np.asarray(dataset.val_imgs).astype(np.float32)
    
    print(train_y.shape, val_y.shape, train_filenames.shape, val_filenames.shape, train_imgs.shape, val_imgs.shape)

    # load models to train
    vgg = models.VGG_model((224, 224, 3))
    vgg.train_model(train_imgs, train_y, val_imgs, val_y)

    
