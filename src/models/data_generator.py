
import numpy as np
import os
import tensorflow as tf
import random
import math

from scipy import ndimage

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, x_set, y_set, root_directory, img_dim, augmentation, shuffle):

        self.batch_size = batch_size
        self.x_set = x_set
        self.y_set = y_set
        self.root_directory = root_directory
        self.img_dim = img_dim
        self.augmentation = augmentation
        self.shuffle = shuffle

        self.indices = np.arange(self.x_set.shape[0])
        self.val_labels = []

    # returns the number of batches to generate
    def __len__(self):
        return len(self.x_set) // self.batch_size
    
    # return a batch of a given index
    # create logic of how to load data
    def __getitem__(self, idx):
        # get batch
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_IDs = self.x_set.iloc[inds]
        batch_labels = self.y_set.iloc[inds]

        images, labels = [], []

        horizontal_shift, vertical_shift, rotation_shift = 0, 0, 0

        for id in range(len(batch_IDs)):
           
            # load image
            path = os.path.join(self.root_directory, batch_IDs.iloc[id, 0])
            image = tf.keras.preprocessing.image.load_img(path, color_mode="rgb", target_size=(self.img_dim, self.img_dim))
            
            # extract augmentations
            if "horizontal shift" in self.augmentation:
                horizontal_shift = (random.randrange(-15, 15)/100)*self.img_dim # up to 20% of width of image
            if "vertical shift" in self.augmentation:
                vertical_shift = (random.randrange(-45, 45)/100)*self.img_dim
            if "rotation" in self.augmentation:
                rotation_shift = (random.randrange(-20, 20))
            if "brightness" in self.augmentation:
                brightness_shift = (random.randrange(-15, 20)/100)
                image = tf.image.adjust_brightness(image, delta=brightness_shift)
            if "contrast" in self.augmentation:
                contrast_shift=(random.randrange(0, 3)/10)
                image = tf.image.adjust_contrast(image, contrast_shift)
            if "colour" in self.augmentation:
                hue_shift=(random.randrange(0,10)/100)
                image = tf.image.adjust_hue(image, hue_shift)

            image = tf.keras.preprocessing.image.img_to_array(image).astype(int)
            # apply augmentations    
            image = ndimage.shift(image, (vertical_shift, horizontal_shift, 0)) # shift is given in pixels
            image = ndimage.rotate(image, rotation_shift, reshape=False)
            images.append(image)
        
            # load labels
            lx = batch_labels.iloc[id, 0]*(self.img_dim/960)+horizontal_shift  # don't remove self.img_dim and replace with 1
            ly = batch_labels.iloc[id, 1]*(self.img_dim/960)+vertical_shift
            rx = batch_labels.iloc[id, 2]*(self.img_dim/960)+horizontal_shift
            ry = batch_labels.iloc[id, 3]*(self.img_dim/960)+vertical_shift

            lx, ly = self.rotate((lx, ly), rotation_shift*(math.pi/180), (self.img_dim/2, self.img_dim/2))
            rx, ry = self.rotate((rx, ry), rotation_shift*(math.pi/180), (self.img_dim/2, self.img_dim/2))

            # get lx, ly, rx, ry relative to size of image
            lx = lx/self.img_dim
            ly = ly/self.img_dim
            rx = rx/self.img_dim
            ry = ry/self.img_dim

            tmp_eye = np.array([lx, ly, rx, ry])
            labels.append(tmp_eye)
            self.val_labels.append(tmp_eye)

        
        # convert list of labels to numpy array0
        labels = np.array(labels)
        images = np.array(images)

        

        return (images, labels)

    def on_epoch_end(self):        
        """ shuffle data at end of each epoch for varied batches """
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def rotate(self, pt, radians, origin):

        x, y = pt
        offset_x, offset_y = origin
        
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        
        cos_rad = math.cos(radians)
        sin_rad = math.sin(radians)
        
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        
        return qx, qy