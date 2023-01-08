import os
import cv2
import datetime
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf

from keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

import config
from build_dataset import build_dataset, partition_dataset

def inspect_data(imgFolder, centres_df, region_df):

    print(centres_df.shape, region_df.shape)

    for row in range(0, centres_df.shape[0]):

        name = centres_df['filename'][row]
        imgFilepath = os.path.join(imgFolder, name)
        print(imgFilepath)

        im = cv2.imread(imgFilepath)

        # draw the labelled centres on the image
        lx = centres_df['lx'][row]
        ly = centres_df['ly'][row]
        rx = centres_df['rx'][row]
        ry = centres_df['ry'][row]
        cv2.circle(im, (lx, ly), 4, (0, 255, 0), -1)
        cv2.circle(im, (rx, ry), 4, (255, 0, 0), -1)

        # get bounding box data that corresponds to the same image
        idx = region_df.index[region_df["filename"] == name]
        startLX = region_df["startLX"].values[idx][0]
        startLY = region_df["startLY"].values[idx][0]
        endLX = region_df["endLX"].values[idx][0]
        endLY = region_df["endLY"].values[idx][0]

        startRX = region_df["startRX"].values[idx][0]
        startRY = region_df["startRY"].values[idx][0]
        endRX = region_df["endRX"].values[idx][0]
        endRY = region_df["endRY"].values[idx][0]

        # draw the predicted bounding box on the image
        cv2.rectangle(im, (startLX, startLY), (endLX, endLY), (0, 255, 0), 2)
        #
        # # draw the predicted bounding box on the image
        cv2.rectangle(im, (startRX, startRY), (endRX, endRY), (255, 0, 0), 2)

        # show the output image
        cv2.imshow("Output", im)
        cv2.waitKey(0)

def inspect_thumbnails(imgFolder, centres_df, region_df):

    left_eye_image = []
    left_eye_targets = []
    left_eye_filenames = []

    all_img_heights = []
    all_img_widths = []

    for row in range(region_df.shape[0]):

        name = region_df['filename'][row]

        imgFilepath = os.path.join(imgFolder, name)
        print(imgFilepath)

        im = cv2.imread(imgFilepath)

        # left eye image
        startLX = region_df["startLX"][row]
        startLY = region_df["startLY"][row]
        endLX = region_df["endLX"][row]
        endLY = region_df["endLY"][row]

        if startLX == 1:
            continue

        new_height = endLY - startLY
        new_width = endLX - startLX

        all_img_widths.append(new_width)
        all_img_heights.append(new_height)

    # determine max image size
    height_max = max(all_img_heights)
    width_max = max(all_img_widths)

    print(height_max, width_max)

    for row in range(region_df.shape[0]):

        name = region_df['filename'][row]

        imgFilepath = os.path.join(imgFolder, name)
        print(imgFilepath)

        im = cv2.imread(imgFilepath)

        # left eye image
        startLX = region_df["startLX"][row]
        startLY = region_df["startLY"][row]
        endLX = region_df["endLX"][row]
        endLY = region_df["endLY"][row]

        if startLX == 1:
            continue

        new_height = endLY - startLY
        new_width = endLX - startLX

        # resize images to squares
        if new_width > new_height:
            # expected condition
            diff = (new_width-new_height)/2

            if diff%2 == 0:
                startLY = int(startLY - diff)
                endLY = int(endLY + diff)
                new_height = int(endLY-startLY)

            if diff%2 == 1:
                pass


        cropped_im = im[startLY:endLY, startLX:endLX]

        idx = region_df.index[region_df["filename"] == name]
        lx = centres_df['lx'].values[idx][0]
        ly = centres_df['ly'].values[idx][0]

        # ignore images labelled incorrectly, or blinks
        if lx == 1:
            continue
        # if rx < 960:
        #     continue

        print(lx, ly)

        cropped_lx = lx - startLX
        cropped_ly = ly - startLY

        # add borders

        print(cropped_lx, cropped_ly)

        left_eye_x = cropped_lx/new_width
        left_eye_y = cropped_ly/new_height

        print(left_eye_x, left_eye_y)

        cv2.circle(cropped_im, (cropped_lx, cropped_ly), 4, (0, 255, 0), -1)
        # cv2.imshow("cropped", cropped_im)
        # cv2.waitKey(0)

        left_eye_targets.append((left_eye_x, left_eye_y))

        # image = load_img(imagePath, target_size=(224,224))
        image = img_to_array(cropped_im)
        left_eye_filenames.append(name)

    cv2.destroyAllWindows()

    return left_eye_image, left_eye_targets, left_eye_filenames

def build_model(model_to_run):

    # if i == 0:
    base_model = None

    if model_to_run == "vgg":
        base_model = VGG16(weights="imagenet", include_top=False,
    	input_tensor=Input(shape=(config.VGG_INPUT_SHAPE, config.VGG_INPUT_SHAPE, 3)))

    if model_to_run == "inception":
        base_model = InceptionV3(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(config.INCEPTIONV3_INPUT_SHAPE, config.INCEPTIONV3_INPUT_SHAPE, 3)))

    # freeze all layers so they will *not* be updated during the
    # training process
    base_model.trainable = False
    # flatten the max-pooling output of VGG
    flatten = base_model.output
    flatten = Flatten()(flatten)
    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(2, activation="sigmoid")(bboxHead)

    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=base_model.input, outputs=bboxHead)


    return model

def load_images_to_test(rootFolder):
    images_to_label = os.path.join(rootFolder, "test.csv")
    df = pd.read_csv(images_to_label)[["filename"]]

    return df

def adjust_contrast(image):

    # contrast factor lower == more white; higher == more dark colours

    return tf.image.adjust_contrast(image, 1.0)

def adjust_hue(image):
    # delta must be in the interval [-1, 1].
    delta = 1
    # -0.5  == blue skin, light brown eyes
    # 0 ==
    # 0.5 ==
    # 1 == normal
    return tf.image.adjust_hue(image, delta, name=None)

def flip_image_and_targets(img, target):

    flipped_img = np.flip(img, 1)
    # print(target)
    target[0] = 1-target[0]
    # print(target)

    return flipped_img, target

def apply_flips(trainImages, trainTargets, save_both=0):

    newTrainImages, newTrainTargets = [], []

    if trainImages.shape[0] != trainTargets.shape[0]:
        print("Number of training samples, and training targets are not equal.")
        return None

    # for every image set 30% chance of flipping
    for i in range(0, trainImages.shape[0]):
        probability = random.random()

        if save_both:
            # save augmented (flipped image), as well as original to dataset (possible duplication?)
            if probability < config.HORIZONTAL_FLIPS:
                # save original image
                newTrainImages.append(trainImages[i])
                newTrainTargets.append(trainTargets[i])

                # flip and save augmented image
                tmp_img, tmp_labels = flip_image_and_targets(trainImages[i], trainTargets[i])
                newTrainTargets.append(tmp_labels)
                newTrainImages.append(tmp_img)

        else:
            if probability < config.HORIZONTAL_FLIPS:
                # flip and save augmented image
                tmp_img, tmp_labels = flip_image_and_targets(trainImages[i], trainTargets[i])
                newTrainTargets.append(tmp_labels)
                newTrainImages.append(tmp_img)
            else:
                # save original image
                newTrainImages.append(trainImages[i])
                newTrainTargets.append(trainTargets[i])


    return newTrainImages, newTrainTargets

def prepare(ds, shuffle=False, augment=False):

    AUTOTUNE = tf.data.AUTOTUNE

    data_augmentation = tf.keras.Sequential([layers.Lambda(adjust_hue), layers.Lambda(adjust_contrast)])

    if shuffle:
        ds = ds.shuffle(42)

    # batch all datasets
    ds = ds.batch(config.BATCH_SIZE)

    # apply data augmentation only on the training set
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

        # for batch_of_images, batch_of_labels in ds.take(20):
        #     count = 0
        #     for image in batch_of_images:
        #         numpy_image = image.numpy()
        #         numpy_labels = batch_of_labels[count].numpy()
        #
        #         x = int(numpy_labels[0]*config.INCEPTIONV3_INPUT_SHAPE)
        #         y = int(numpy_labels[1]*config.INCEPTIONV3_INPUT_SHAPE)
        #
        #         cv2.circle(numpy_image, (x, y), 4, (0, 255, 0), -1)
        #         cv2.imshow("image", numpy_image)
        #         cv2.waitKey(0)
        #         count += 1
        #
        #     break

    return ds.prefetch(buffer_size=AUTOTUNE)

if __name__ == '__main__':

    # retrieve bounding box and eye centre coordinates
    bdbxDirectory = os.path.join(config.ROOTFOLDER.replace("improved_ECL", "eye_region_detector"), "output", "eye_region")# config.PARTICIPANT_CSV_FILE)
    bdbxFilepaths = [os.path.join(bdbxDirectory, csv_file) for csv_file in os.listdir(bdbxDirectory)]

    region_df = pd.read_csv(bdbxFilepaths[0])[['filename', 'startLX', 'startLY', 'endLX', 'endLY', 'startRX', 'startRY', 'endRX', 'endRY']]

    for i in range(1, len(bdbxFilepaths)):
        tmp_df = pd.read_csv(bdbxFilepaths[i])[['filename', 'startLX', 'startLY', 'endLX', 'endLY', 'startRX', 'startRY', 'endRX', 'endRY']]
        region_df = pd.concat([region_df, tmp_df], axis = 0)

    eyeCentreDirectory = os.path.join(config.ROOTFOLDER.replace("improved_ECL", "eye_centre_localisation"), "labels_eme2")#, config.PARTICIPANT_CSV_FILE)
    eyeCentreFilepaths = [os.path.join(eyeCentreDirectory, csv_file) for csv_file in os.listdir(eyeCentreDirectory)]

    centres_df = pd.read_csv(eyeCentreFilepaths[0])[['filename', 'lx', 'ly', 'rx', 'ry']]

    for i in range(1, len(eyeCentreFilepaths)):
        tmp_df = pd.read_csv(eyeCentreFilepaths[i])[['filename', 'lx', 'ly', 'rx', 'ry']]
        centres_df = pd.concat([centres_df, tmp_df], axis=0)

    print("Images with labelled bounding boxes: " + str(region_df.shape[0]))
    print("Images with labelled eye centres: " + str(centres_df.shape[0]))

    # inspect_data(imgFolder, centres_df, region_df)
    # image, target, filename = inspect_thumbnails(imgFolder, centres_df, region_df)

    # BUILD DATASET
    data, targets, filenames = build_dataset(config.IMGFOLDER, centres_df, region_df, config.INCEPTIONV3_INPUT_SHAPE)
    print(len(data), len(targets))

    # build train, validation, and test split using left and right eyes
    trainImages, testImages, trainTargets, testTargets, trainFilenames, testFilenames = partition_dataset(data, targets, filenames)
    print(len(trainImages), len(trainTargets))


    # apply horizontal flips to trainImages and appropriate transformation to corresponding trainTargets
    # print("[INFO] Applying horizontal flips at random to training dataset...")
    # trainImages, trainTargets = apply_flips(trainImages, trainTargets)

    print("Training Images: " + str(len(trainImages)))
    print("Testing Images: " + str(len(testImages)))

    # convert existing training/validation/test sets to tensorflow dataset
    train_ds = tf.data.Dataset.from_tensor_slices((trainImages, trainTargets))
    test_ds = tf.data.Dataset.from_tensor_slices((testImages, testTargets))

    # apply relevant data augmentation to dataset
    train_ds = prepare(train_ds, shuffle=False, augment=True)
    test_ds = prepare(test_ds, shuffle=False, augment=False)

    ###### TRAIN MODEL
    # test performance of different models
    for i in range(0, 1):
        modelname = "inception"
        # load and set up model
        model = build_model(modelname)
        opt = Adam(learning_rate=config.INIT_LR)
        model.compile(loss="mse", optimizer=opt)
        # print(model.summary())

        # Callbacks - recording the intermediate training results which can be visualised on tensorboard
        subFolderLog = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboardPath = os.path.join(os.getcwd(), "output", "logs", subFolderLog)
        checkpointPath = os.path.join(os.getcwd(), "output", "logs" , subFolderLog)
        checkpointPath = checkpointPath + "/" + modelname +"-weights-improve-{epoch:02d}-{val_loss:02f}.hdf5"

        callbacks = [
            TensorBoard(log_dir=tensorboardPath, histogram_freq=0, write_graph=True, write_images=True),
            ModelCheckpoint(filepath=checkpointPath, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        ]

        # train the network for bounding box regression
        print("[INFO] training eye centre localisation model...")
        H = model.fit(
        	train_ds,
            validation_data=test_ds,
        	epochs=config.NUM_EPOCHS,
        	verbose=1,
            callbacks=callbacks)

        # model.evaluate(test_ds)

        # plot the model training history
        N = config. NUM_EPOCHS
        # plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.title("Eye Centre Localisation Loss on Training Set")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        PLOT_PATH = os.path.join(os.getcwd(), "output", str(modelname) + "_plot.png")
        plt.savefig(PLOT_PATH)
