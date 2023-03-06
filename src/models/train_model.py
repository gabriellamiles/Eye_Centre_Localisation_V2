"Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory"

import os
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

from keras.callbacks import TensorBoard, ModelCheckpoint

from tensorflow.keras.optimizers import Adam

import model_utils

def load_csv_files_from_directory(directory):

    all_df = []

    for i in os.listdir(directory):

        full_filepath = os.path.join(directory, i)

        if i[-4:] == ".csv":
            tmp_df = pd.read_csv(full_filepath)[["filename", "LE_left", "LE_top", "LE_right", "LE_bottom", "RE_left", "RE_top", "RE_right", "RE_bottom", "lx", "ly", "rx", "ry"]]
            all_df.append(tmp_df)

    return all_df

def resize_image_as_array(im, new_dim, targets, show_images=0):

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

def initialise_dataset(img_folder, list_of_label_dfs, standard_size):

    data, targets, filenames = [], [], []

    for trial in list_of_label_dfs:

        # print(trial)

        for i in range(trial.shape[0]):
            filename = trial["filename"][i]
    #
            for individual_eye_folder in img_folder:
                image_path = os.path.join(individual_eye_folder, filename)
                # print(image_path)
    #
                try:
                    im = Image.open(image_path)


                    if "left_eye" in individual_eye_folder:
                        relative_lx = trial["relative_lx"][i]
                        relative_ly = trial["relative_ly"][i]

                        if relative_lx < 0 or relative_ly<0:
                            break

                        filenames.append(os.path.join("left_eye",filename))
                        img_as_array, updated_labels = resize_image_as_array(im, standard_size, [relative_lx, relative_ly])
                        data.append(img_as_array)
                        targets.append(updated_labels)

                    elif "right_eye" in individual_eye_folder:
                        relative_rx = trial["relative_rx"][i]
                        relative_ry = trial["relative_ry"][i]

                        if relative_rx < 0 or relative_ry<0:
                            break

                        filenames.append(os.path.join("right_eye",filename))
                        img_as_array, updated_labels = resize_image_as_array(im, standard_size, [relative_rx, relative_ry])
                        data.append(img_as_array)
                        targets.append(updated_labels)

                    im.close() # remember to close image

                except FileNotFoundError:
                    # print("Directory likely still loading...")
                    pass

    return data, targets, filenames

def partition_dataset(data, targets, filenames, test_image_folder):

    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")

    print(data.shape, targets.shape)


    # partition the data into training and testing splits using 90% of the data for
    # training and remaining 10% for testing
    X_train, X_test, y_train, y_test, X_filenames, y_filenames = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)


    return X_train, X_test, y_train, y_test, X_filenames, y_filenames

def get_corresponding_labels(labels_folder, img_folder):

    corresponding_filepaths = []

    labels_filepaths = get_csv_filepaths_from_directory(labels_folder)

    # print(img_folder)
    # print(labels_filepaths)

    for folder in img_folder:
        new_replacement = folder.split("/processed/")[-1]

        for filepath in labels_filepaths:
            tmp_filepath = filepath.replace(".csv", "") # get rid of csv extension for each file
            tmp_filepath = tmp_filepath.replace("combined_labels", new_replacement) # path to directory that corresponding images are stored at
            tmp_filepath = tmp_filepath.rsplit("_",1) # change 003_0 to 003/0 for example

            full_filepath = ''
            for item in tmp_filepath:
                full_filepath = os.path.join(full_filepath, item)

            if not os.path.exists(full_filepath):
                # print(full_filepath)
                continue

            corresponding_filepaths.append(filepath)

    return corresponding_filepaths

def load_csv_files_from_list(labels_filepaths):

    list_of_df = []

    for filepath in labels_filepaths:
        df = pd.read_csv(filepath)[["filename", "LE_left", "LE_top", "LE_right", "LE_bottom", "RE_left", "RE_top", "RE_right", "RE_bottom","relative_lx", "relative_ly", "relative_rx", "relative_ry"]]
        list_of_df.append(df)

    return list_of_df

def get_csv_filepaths_from_directory(directory):

    list_of_filepaths = []

    for file in os.listdir(directory):
        if file[-4:] == ".csv":
            list_of_filepaths.append(os.path.join(directory, file))

    return list_of_filepaths

def get_max_image_size(labels_df):

    max_height_all_df = []
    max_width_all_df = []

    for df in labels_df:

        width_left = (df["LE_right"] - df["LE_left"]).max()
        width_right = (df["RE_right"] - df["RE_left"]).max()
        max_width = max([width_left, width_right])

        height_left = (df["LE_bottom"] - df["LE_top"]).max()
        height_right = (df["RE_bottom"] - df["RE_top"]).max()
        max_height = max([height_left, height_right])

        max_width_all_df.append(max_width)
        max_height_all_df.append(max_height)

    return max(max_width_all_df), max(max_height_all_df)

def save_test_set_information(testImages, testFilenames, testTargets, test_image_folder, new_size, save_test_images=0):

    filenamesToSave = pd.Series(testFilenames, name="filename")
    testTargetsToSave = pd.DataFrame(testTargets, columns = ["x", "y"])*224# multiplied by size of image
    testDataToSave = pd.concat([filenamesToSave, testTargetsToSave], axis=1)
    testDataToSave.to_csv(os.path.join(test_image_folder, "labels.csv"))

    if save_test_images:
        print("Saving test images:")
        count = 0
        for image in testImages:
            # print(image)
            save_im = Image.fromarray((image*255).astype(np.uint8), "RGB")
            save_directories = filenamesToSave[count].rsplit("/",1)[0]

            save_image_directory = os.path.join(test_image_folder, "imgs", save_directories)

            if not os.path.exists(save_image_directory):
                os.makedirs(save_image_directory)

            save_filepath = os.path.join(test_image_folder, "imgs", filenamesToSave[count])
            save_im.save(save_filepath)

            count+=1
        print("Test information saved.")


if __name__ == '__main__':

    # key hyperparameters
    BATCH_SIZE=32
    NUM_EPOCHS=20

    # initialise key filepaths
    root_folder = os.getcwd()
    img_root_folder = os.path.join(root_folder, "data", "processed", "mnt", "cropped_eye_imgs")

    img_folders =  []
    for eye_folder in os.listdir(img_root_folder):
        if os.path.isdir(os.path.join(img_root_folder, eye_folder)):
            img_folders.append(os.path.join(img_root_folder, eye_folder))

    label_folder = os.path.join(root_folder, "data", "processed", "combined_labels")
    test_image_folder = os.path.join(root_folder, "data", "processed", "test_data")

    print("[INFO] Initialise key dataset features...")
    labels_filepaths = get_corresponding_labels(label_folder, img_folders) # return labels that there are existing images for
    print("Number of label dataframes loaded: " + str(len(labels_filepaths)))
    labels_df = load_csv_files_from_list(labels_filepaths)

    # determine maximum image size
    w, h = get_max_image_size(labels_df)
    print("Max image size (w, h):" , str(w), str(h))

    print("[INFO] Loading dataset...")
    # build dataset (w images) and ensure all images in dataset are of equivalent size
    data, targets, filenames = initialise_dataset(img_folders, labels_df, max([w,h]))
    data = data[:15000]
    targets = targets[:15000]
    filenames = filenames[:15000]
    print(len(data), len(targets), len(filenames))

    # split into training/testing/validation data
    print("[INFO] Create train/test splits...")
    trainImages, testImages, trainTargets, testTargets, trainFilenames, testFilenames = partition_dataset(data, targets, filenames, test_image_folder)
    save_test_set_information(testImages, testFilenames, testTargets, test_image_folder, w, save_test_images=1)

    # print("[INFO] Loading model...")
    # model, keyword = model_utils.load_model((224, 224, 3))
    # opt = Adam(lr=1e-4)
    # model.compile(loss="mse", optimizer=opt)
    #
    # ###############################################
    # # Callbacks - recording the intermediate training results which can be visualised on tensorboard
    # subFolderLog = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # tensorboardPath = os.path.join(os.getcwd(), "models", subFolderLog)
    # checkpointPath = os.path.join(os.getcwd(), "models" , subFolderLog)
    # checkpointPath = checkpointPath + "/" + keyword + "-weights-improve-{epoch:02d}-{val_loss:02f}.hdf5"
    #
    # callbacks = [
    #     TensorBoard(log_dir=tensorboardPath, histogram_freq=0, write_graph=True, write_images=True),
    #     ModelCheckpoint(filepath=checkpointPath, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
    # ]
    #
    # # train the network for bounding box regression
    # print("[INFO] training eye centre regressor...")
    # H = model.fit(
    #     trainImages, trainTargets,
    #     validation_data=(testImages, testTargets),
    #     batch_size=BATCH_SIZE,
    #     epochs=NUM_EPOCHS,
    #     callbacks=callbacks,
    #     verbose=1)
    #
    # # plot the model training history
    # plt.figure()
    # plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
    # plt.title("Eye Centre Regression Loss on Training Set")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss")
    # plt.legend(loc="lower left")
    # save_path = os.path.join(root_folder, tensorboardPath, keyword + "training_plot.png")
    # plt.savefig(save_path)
