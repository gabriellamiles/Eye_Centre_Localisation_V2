
import numpy as np
import os
import tensorflow as tf
import random
import time
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

        full_start = time.time()

        for id in range(len(batch_IDs)):
            # load image
            path = os.path.join(self.root_directory, batch_IDs.iloc[id, 0])
            try:
                image = tf.keras.preprocessing.image.load_img(path, color_mode="rgb", target_size=(self.img_dim, self.img_dim))
                image = tf.keras.preprocessing.image.img_to_array(image).astype(int)
                # load_img_time = time.time()
                
                # extract augmentations
                # if "horizontal shift" in self.augmentation:
                #     horizontal_shift = (random.randrange(-15, 15)/100)*self.img_dim # up to 20% of width of image
                # if "vertical shift" in self.augmentation:
                #     vertical_shift = (random.randrange(-45, 45)/100)*self.img_dim
                # if "rotation" in self.augmentation:
                #     rotation_shift = (random.randrange(-20, 20))
                
                # check_augmentations_ti/me = time.time()
                # if "brightness" in self.augmentation:
                #     brightness_shift = (random.randrange(-5, 5)/100)
                #     image = tf.image.adjust_brightness(image, delta=brightness_shift)
                # if "contrast" in self.augmentation:
                #     contrast_shift=(random.randrange(0, 3)/10)
                #     image = tf.image.adjust_contrast(image, contrast_shift)
                # if "colour" in self.augmentation:
                #     hue_shift=(random.randrange(0,10)/100)
                #     image = tf.image.adjust_hue(image, hue_shift)

                
                
                ###### THIS SECTION TAKES ABOUT 90% OF THE TIME
                # apply augmentations
                # if "horizontal shift" in self.augmentation:
                #     image = ndimage.shift(image, (0, horizontal_shift, 0)) # shift is given in pixels
                # if "vertical shift" in self.augmentation:
                #     image = ndimage.shift(image, (vertical_shift, 0, 0)) # shift is given in pixels
                # if "rotation" in self.augmentation:
                #     image = ndimage.rotate(image, rotation_shift, reshape=False)
                ##### ABOVE SECTION TAKES ABOUT 90% OF THE TIME
                
                images.append(image)
                # apply_aug_time = time.time()
                
            
                # load labels
                lx = batch_labels.iloc[id, 0]*(self.img_dim/960)+horizontal_shift  # don't remove self.img_dim and replace with 1
                ly = batch_labels.iloc[id, 1]*(self.img_dim/960)+vertical_shift
                rx = batch_labels.iloc[id, 2]*(self.img_dim/960)+horizontal_shift
                ry = batch_labels.iloc[id, 3]*(self.img_dim/960)+vertical_shift

                # lx, ly = self.rotate((lx, ly), rotation_shift*(math.pi/180), (self.img_dim/2, self.img_dim/2))
                # rx, ry = self.rotate((rx, ry), rotation_shift*(math.pi/180), (self.img_dim/2, self.img_dim/2))

                # get lx, ly, rx, ry relative to size of image
                lx = lx/self.img_dim
                ly = ly/self.img_dim
                rx = rx/self.img_dim
                ry = ry/self.img_dim

                tmp_eye = np.array([lx, ly, rx, ry])
                labels.append(tmp_eye)
                self.val_labels.append(tmp_eye)

                # print("load img: " + str(load_img_time-start))
                # print("check aug time: " + str(check_augmentations_time-load_img_time))
                # print("apply aug time: " + str(apply_aug_time-check_augmentations_time))
                # print("update labels time: " + str(update_labels_time-apply_aug_time))
                
                # print("full processing one im: " + str(end_one_img-start))
            except:
                print("Image not found... " + str(path))
                pass

        
        # convert list of labels to numpy array0
        labels = np.array(labels)
        images = np.array(images)

        # print("full batch processing time: " + str(end_one_batch- full_start))
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
    
class SingleEyeDataGenerator(tf.keras.utils.Sequence):
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

        for id in range(len(batch_IDs)):

            # load image
            path = os.path.join(self.root_directory, batch_IDs.iloc[id, 0])
            image = tf.keras.preprocessing.image.load_img(path, color_mode="rgb", target_size=(self.img_dim, self.img_dim))
            image = tf.keras.preprocessing.image.img_to_array(image).astype(int)
            images.append(image)
            
        
            # load labels
            x = batch_labels.iloc[id, 0]#*(self.img_dim/299)+horizontal_shift  # don't remove self.img_dim and replace with 1
            y = batch_labels.iloc[id, 1]#*(self.img_dim/299)+vertical_shift


            # get lx, ly, rx, ry relative to size of image
            x = x/self.img_dim
            y = y/self.img_dim

            tmp_eye = np.array([x, y])
            labels.append(tmp_eye)
            self.val_labels.append(tmp_eye)

            # print("load img: " + str(load_img_time-start))
            # print("check aug time: " + str(check_augmentations_time-load_img_time))
            # print("apply aug time: " + str(apply_aug_time-check_augmentations_time))
            # print("update labels time: " + str(update_labels_time-apply_aug_time))

        
        # convert list of labels to numpy array0
        labels = np.array(labels)
        images = np.array(images)


        # print("full batch processing time: " + str(end_one_batch- full_start))
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
    
class UnseenDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, x_set, root_directory, img_dim):

        self.batch_size = batch_size
        self.x_set = x_set
        self.root_directory = root_directory
        self.img_dim = img_dim


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
        # batch_labels = self.y_set.iloc[inds]

        images = []

        for id in range(len(batch_IDs)):

            # load image
            path = os.path.join(self.root_directory, batch_IDs.iloc[id, 0])
            image = tf.keras.preprocessing.image.load_img(path, color_mode="rgb", target_size=(self.img_dim, self.img_dim))
            image = tf.keras.preprocessing.image.img_to_array(image).astype(int)
            images.append(image)
            
        
            # load labels
            # x = batch_labels.iloc[id, 0]#*(self.img_dim/299)+horizontal_shift  # don't remove self.img_dim and replace with 1
            # y = batch_labels.iloc[id, 1]#*(self.img_dim/299)+vertical_shift


            # get lx, ly, rx, ry relative to size of image
            # x = x/self.img_dim
            # y = y/self.img_dim

            # tmp_eye = np.array([x, y])
            # labels.append(tmp_eye)
            # self.val_labels.append(tmp_eye)

            # print("load img: " + str(load_img_time-start))
            # print("check aug time: " + str(check_augmentations_time-load_img_time))
            # print("apply aug time: " + str(apply_aug_time-check_augmentations_time))
            # print("update labels time: " + str(update_labels_time-apply_aug_time))

        
        # convert list of labels to numpy array0
        # labels = np.array(labels)
        images = np.array(images)


        # print("full batch processing time: " + str(end_one_batch- full_start))
        return images