import os

import numpy as np
import config
from sklearn.model_selection import train_test_split, KFold
from PIL import Image
from keras.utils import load_img, img_to_array


class Dataset():
    def __init__(self,
                 labels=None
                 ):
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None

        self.get_eye_centres()
        self.get_left_eye_centres()
        self.get_right_eye_centres()
        
    def get_eye_centres(self):
        """ Returns absolute eye centres."""
        self.eye_centres = self.labels[["filename", "lx", "ly", "rx", "ry"]]

    def get_left_eye_centres(self):
        """Returns only left eye centres (relative to bounding box)"""
        self.left_eye_centres = self.labels[["filename", "relative_lx", "relative_ly"]]
    
    def get_right_eye_centres(self):
        """Returns only right eye centres (relative to bounding box)"""
        self.right_eye_centres = self.labels[["filename", "relative_rx", "relative_ry"]]

    def get_train_val_test(self, labels=None):
        """ Function splits labels into train/validation/test labels. """

        if labels is None:
            labels = self.labels
        # train/test/validation split -- 0.6/0.2/0.2
        self.train_labels, self.test_labels = train_test_split(labels, test_size=0.2) # split into test and train
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
    
    def get_cropped_images(self, set_of_labels):

        resized_imgs, updated_labels = [], []
        
        for row in range(set_of_labels.shape[0]):


            filepath = set_of_labels.iloc[row, 0]
            targets = (set_of_labels.iloc[row, 1], set_of_labels.iloc[row, 2])

            full_im_filepath = os.path.join(config.cropped_img_folder, filepath)
            im = Image.open(full_im_filepath)

            resized_im, updated_targets = self.resize_image_as_array(im, 224, targets)
            resized_imgs.append(resized_im)
            updated_labels.append(updated_targets)

        return resized_imgs, updated_labels


    def resize_image_as_array(self, im, new_dim, targets, show_images=0):

        old_size = im.size # determine size of original imag
        new_size = (new_dim, new_dim)
        new_im = Image.new("RGB", new_size)
        box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))

        new_im.paste(im, box)
        x = targets[0] + int((new_size[0] - old_size[0])/2)
        y = targets[1] + int((new_size[1] - old_size[1])/2)

        # get x and y as ratio of image
        x = x/new_dim
        y = y/new_dim

        if 224 < new_dim :
            # resize images for ml input
            new_im = new_im.resize((224, 224))

        if show_images:
            draw = ImageDraw.Draw(new_im)
            draw.regular_polygon((int(x*224), int(224), 10), 4, 0, fill=(0, 255, 0), outline=(0, 255, 0))
            new_im.show()

        img_as_array = img_to_array(new_im)
        new_im.close()

        return img_as_array, [x, y]

    def save_test_data(self):
        """ Saves csv file containing test split (filenames + targets)."""        
        self.test_labels.to_csv(config.test_split_save_location)