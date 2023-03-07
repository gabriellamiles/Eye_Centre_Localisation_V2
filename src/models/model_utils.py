import os

import pandas as pd

import config

def load_model(size=None):

    if size == None:
        size = (224, 224,3)

    vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=size))
    # freeze all VGG layers so they will *not* be updated during the
    # training process
    vgg.trainable = False
    # flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)
    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(2, activation="sigmoid")(bboxHead)
    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=vgg.input, outputs=bboxHead)
    keyword = 'vgg'

    return model, keyword

def load_images(img_root_folder):
    img_folders =  []
    for eye_folder in os.listdir(img_root_folder):
        if os.path.isdir(os.path.join(img_root_folder, eye_folder)):
            img_folders.append(os.path.join(img_root_folder, eye_folder))

    return img_folders

def load_csv_files_from_list(labels_filepaths, columns=["filename", "LE_left", "LE_top", "LE_right", "LE_bottom", "RE_left", "RE_top", "RE_right", "RE_bottom", "lx", "ly", "rx", "ry", "relative_lx", "relative_ly", "relative_rx", "relative_ry"]):

    list_of_df = []

    for filepath in labels_filepaths:
        df = pd.read_csv(filepath)[columns]
        list_of_df.append(df)

    return list_of_df

def get_csv_filepaths_from_directory(directory):

    list_of_filepaths = []

    for file in os.listdir(directory):
        if file[-4:] == ".csv":
            list_of_filepaths.append(os.path.join(directory, file))

    return list_of_filepaths

def combine_list_into_single_df(list_of_df):

    all_df = list_of_df[0] # intialise dataframe with first dataframe in list

    for df in list_of_df:
        all_df = pd.concat([all_df, df], axis=0)

    return all_df

def load_labels():

    # load labels
    label_filepaths = get_csv_filepaths_from_directory(config.label_folder)
    list_label_df = load_csv_files_from_list(label_filepaths)
    combined_df = combine_list_into_single_df(list_label_df)

    return combined_df