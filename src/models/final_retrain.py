import os
import datetime
import math

import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split

import config
from data_generator import CustomDataGenerator

def load_csv_files(directory):

    all_csv_filepaths = os.listdir(directory)

    all_df = []

    for filepath in all_csv_filepaths:

        if filepath[-4:] == ".csv":
            tmp_df = pd.read_csv(os.path.join(directory, filepath))[["filename", "lx", "ly", "rx", "ry"]]
            all_df.append(tmp_df)

    return all_df

def combine_into_single_dataframe(all_df):

    combined_df = all_df[0] # initialise

    for i in range(1, len(all_df)): # skip the first dataframe as it is already in list
        combined_df = pd.concat((combined_df, all_df[i]))

    return combined_df

def load_xception_model(ending_layers):

    inputs = Input(shape=(299, 299, 3))
    
    xception = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3)
    )
    xception.trainable = False

    preprocess_layers = x = tf.keras.applications.xception.preprocess_input(inputs)
    base_model = xception(preprocess_layers, training=False)

    if ending_layers == 0: 
        flatten = Flatten()(base_model)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="linear")(bboxHead)
        model = Model(inputs=inputs, outputs=bboxHead)
        keyword = 'xception_ending0_retrain'
        return model, keyword
    elif ending_layers == 1:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'xception_ending1_retrain'
        return model, keyword
    elif ending_layers == 2:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(128, activation="relu")(x)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'xception_ending2_retrain'
        return model, keyword

def load_inceptionresnetv2_model(ending_layers):

    inputs = Input(shape=(299, 299, 3))

    inceptionresnetv2 = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3)
    )
   
    inceptionresnetv2.trainable = False

    preprocess_layers = x = tf.keras.applications.inception_resnet_v2.preprocess_input(inputs)
    base_model = inceptionresnetv2(preprocess_layers, training=False)

    if ending_layers == 0:
        flatten = Flatten()(base_model)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="linear")(bboxHead)
        model = Model(inputs=inputs, outputs=bboxHead)
        keyword = 'inceptionresnetv2_ending0_retrain'
        return model, keyword
    elif ending_layers==1:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'inceptionresnetv2_ending1_retrain'
        return model, keyword
    elif ending_layers == 2:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(128, activation="relu")(x)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'inceptionresnetv2_ending2_retrain'
        return model, keyword

def load_mobilenetv2_model(ending_layers):

    inputs = Input(shape=(224, 224, 3))


    mobilenetv2 = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=False,
        weights="imagenet"
    )

    preprocess_layers = x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    base_model = mobilenetv2(preprocess_layers, training=False)

    mobilenetv2.trainable = False
    if ending_layers == 0:
        flatten = Flatten()(base_model)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="linear")(bboxHead)
        model = Model(inputs=inputs, outputs=bboxHead)
        keyword = 'mobilenetv2_ending0_retrain'
        return model, keyword
    elif ending_layers==1:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'mobilenetv2_ending1_retrain'
        return model, keyword
    elif ending_layers == 2:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(128, activation="relu")(x)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'mobilenetv2_ending2_retrain'
        return model, keyword

def load_inception_model(ending_layers):

    inputs = Input(shape=(299, 299, 3))

    inceptionv3 = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3)
    )
   
    inceptionv3.trainable = False

    preprocess_layers = x = tf.keras.applications.inception_v3.preprocess_input(inputs)
    base_model = inceptionv3(preprocess_layers, training=False)

    if ending_layers == 0:
        flatten = Flatten()(base_model)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="linear")(bboxHead)
        model = Model(inputs=inputs, outputs=bboxHead)
        keyword = 'inceptionv3_ending0'
        return model, keyword
    elif ending_layers==1:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'inceptionv3_ending1'
        return model, keyword
    elif ending_layers == 2:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(128, activation="relu")(x)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'inceptionv3_ending2'
        return model, keyword
    
def load_vgg_model(ending_layers):

    inputs = Input(shape=(299, 299, 3))

    vgg16 = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3)
    )
   
    vgg16.trainable = False

    preprocess_layers = x = tf.keras.applications.vgg16.preprocess_input(inputs)
    base_model = vgg16(preprocess_layers, training=False)

    if ending_layers == 0:
        flatten = Flatten()(base_model)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="linear")(bboxHead)
        model = Model(inputs=inputs, outputs=bboxHead)
        keyword = 'vgg16_ending0'
        return model, keyword
    elif ending_layers==1:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'vgg16_ending1'
        return model, keyword
    elif ending_layers == 2:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(128, activation="relu")(x)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'vgg16_ending2'
        return model, keyword
    
def load_resnet152v2_model(ending_layers):

    inputs = Input(shape=(299, 299, 3))

    resnet152v2 = tf.keras.applications.ResNet152V2(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3)
    )
   
    resnet152v2.trainable = False

    preprocess_layers = x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    base_model = resnet152v2(preprocess_layers, training=False)

    if ending_layers == 0:
        flatten = Flatten()(base_model)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="linear")(bboxHead)
        model = Model(inputs=inputs, outputs=bboxHead)
        keyword = 'resnetv2_ending0'
        return model, keyword
    elif ending_layers==1:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'resnetv2_ending1'
        return model, keyword
    elif ending_layers == 2:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(128, activation="relu")(x)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'resnetv2_ending2'
        return model, keyword
    
def load_efficientnetb6_model(ending_layers):

    inputs = Input(shape=(299, 299, 3))

    efficientnetb6 = tf.keras.applications.EfficientNetB6(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3)
    )
   
    efficientnetb6.trainable = False

    preprocess_layers = x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    base_model = efficientnetb6(preprocess_layers, training=False)

    if ending_layers == 0:
        flatten = Flatten()(base_model)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="linear")(bboxHead)
        model = Model(inputs=inputs, outputs=bboxHead)
        keyword = 'efficientnetb6_ending0'
        return model, keyword
    elif ending_layers==1:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'efficientnetb6_ending1'
        return model, keyword
    elif ending_layers == 2:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(128, activation="relu")(x)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'efficientnetb6_ending2'
        return model, keyword

def load_convnext_model(ending_layers):

    inputs = Input(shape=(299, 299, 3))

    convnext = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3)
    )
   
    convnext.trainable = False

    preprocess_layers = x = tf.keras.applications.convnext.preprocess_input(inputs)
    base_model = convnext(preprocess_layers, training=False)

    if ending_layers == 0:
        flatten = Flatten()(base_model)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="linear")(bboxHead)
        model = Model(inputs=inputs, outputs=bboxHead)
        keyword = 'convnext_ending0'
        return model, keyword
    elif ending_layers==1:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'convnext_ending1'
        return model, keyword
    elif ending_layers == 2:
        x = GlobalAveragePooling2D()(base_model)
        x = Dense(128, activation="relu")(x)
        x = Dense(4, activation="linear")(x)
        model = Model(inputs=inputs, outputs=x)
        keyword = 'convnext_ending2'
        return model, keyword

def calculate_visual_angle_error(predictions, labels):

    # print("Predictions...")
    # print(predictions)
    # print("Labels...")
    # print(labels)

    error = np.absolute(predictions - labels)

    # get angle error (in degrees)
    angle_error = np.arctan((error*(609/1920))/400)*(180/math.pi) # 609 = screen width, mm; 1920 = screen res, pxels; 400 = camera dist, mm

    # print("Error: ")
    # print(error)
    # print(angle_error)

    mean_angle_error = np.mean(angle_error, axis=0)
    overall_mean_error = np.mean(mean_angle_error)

    print("Visual angle errors...")
    print(mean_angle_error)
    print(overall_mean_error)

if __name__=='__main__':

    # key information
    root_folder = os.getcwd()
    train_img_folder = os.path.join(root_folder, "data", "final_dataset", "images")

    # obtain labels
    eye_centre_data_folder = os.path.join(root_folder, "data", "processed", "combined_labels")
    list_of_eye_centre_labels = load_csv_files(eye_centre_data_folder)
    eye_centre_df = combine_into_single_dataframe(list_of_eye_centre_labels)

    # remove blinks from labels
    print(eye_centre_df.shape)
    eye_centre_df = eye_centre_df[(eye_centre_df["lx"]>50) & (eye_centre_df["ly"]>50)]
    eye_centre_df = eye_centre_df[eye_centre_df["filename"].str.contains("018")==False]

    print(eye_centre_df.shape)

    # split into train and test sets
    train_eye_centre_df, test_eye_centre_df = train_test_split(eye_centre_df, test_size=0.20, random_state=42, shuffle=True)
    train_eye_centre_df, val_eye_centre_df = train_test_split(train_eye_centre_df, test_size=0.20, random_state=42, shuffle=True)
    print(train_eye_centre_df.shape, val_eye_centre_df.shape, test_eye_centre_df.shape)

    # define data generator parameters
    img_height, img_width = 299, 299
    batch_size = 32

    # set up data generators
    train_ds = CustomDataGenerator(
        batch_size=32,
        x_set=train_eye_centre_df[["filename"]],
        y_set=train_eye_centre_df[["lx", "ly", "rx", "ry"]],
        root_directory=train_img_folder,
        img_dim=img_height,
        augmentation=None,
        shuffle=True
    )

    val_ds = CustomDataGenerator(
        batch_size=32,
        x_set=val_eye_centre_df[["filename"]],
        y_set=val_eye_centre_df[["lx", "ly", "rx", "ry"]],
        root_directory=train_img_folder,
        img_dim=img_height,
        augmentation=None,
        shuffle=True
    )

    # load and compile model
    print("[INFO] Building and compiling model...")

    accuracy_df = []

    for test in range(0,24):

        # currently running tests: 6-9 on gpu

        model, keyword = '', ''
        model_directory = ""
        weights_filepath = ""

        if test == 0:
            model_directory = os.path.join(root_folder, "models", config.model_directory_inception_resnet_0) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_inceptionresnetv2_model(ending_layers=0) # edit this for different models

        elif test == 1:
            model_directory = os.path.join(root_folder, "models", config.model_directory_inception_resnet_1) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_inceptionresnetv2_model(ending_layers=1) # edit this for different models

        elif test == 2:
            model_directory = os.path.join(root_folder, "models", config.model_directory_inception_resnet_2) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_inceptionresnetv2_model(ending_layers=2) # edit this for different models
        
        elif test == 3:

            model_directory = os.path.join(root_folder, "models", config.model_directory_xception_0) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_xception_model(ending_layers=0) # edit this for different models

        elif test == 4:

            model_directory = os.path.join(root_folder, "models", config.model_directory_xception_1) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_xception_model(ending_layers=1) # edit this for different models
    
        elif test == 5:

            model_directory = os.path.join(root_folder, "models", config.model_directory_xception_2) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_xception_model(ending_layers=2) # edit this for different models
    
        elif test == 6:
            # define data generator parameters
            img_height = 224

            # set up data generators
            train_ds = CustomDataGenerator(
                batch_size=32,
                x_set=train_eye_centre_df[["filename"]],
                y_set=train_eye_centre_df[["lx", "ly", "rx", "ry"]],
                root_directory=train_img_folder,
                img_dim=img_height,
                augmentation=None,
                shuffle=True
            )

            val_ds = CustomDataGenerator(
                batch_size=32,
                x_set=val_eye_centre_df[["filename"]],
                y_set=val_eye_centre_df[["lx", "ly", "rx", "ry"]],
                root_directory=train_img_folder,
                img_dim=img_height,
                augmentation=None,
                shuffle=True
            )
            model_directory = os.path.join(root_folder, "models", config.model_directory_mobilenet_0) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            
            model, keyword = load_mobilenetv2_model(ending_layers=0) # edit this for different models

        elif test == 7:
            # define data generator parameters
            img_height = 224

            # set up data generators
            train_ds = CustomDataGenerator(
                batch_size=32,
                x_set=train_eye_centre_df[["filename"]],
                y_set=train_eye_centre_df[["lx", "ly", "rx", "ry"]],
                root_directory=train_img_folder,
                img_dim=img_height,
                augmentation=None,
                shuffle=True
            )

            val_ds = CustomDataGenerator(
                batch_size=32,
                x_set=val_eye_centre_df[["filename"]],
                y_set=val_eye_centre_df[["lx", "ly", "rx", "ry"]],
                root_directory=train_img_folder,
                img_dim=img_height,
                augmentation=None,
                shuffle=True
            )
            model_directory = os.path.join(root_folder, "models", config.model_directory_mobilenet_1) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            
            model, keyword = load_mobilenetv2_model(ending_layers=1) # edit this for different models

        elif test == 8:
            # define data generator parameters
            img_height = 224

            # set up data generators
            train_ds = CustomDataGenerator(
                batch_size=32,
                x_set=train_eye_centre_df[["filename"]],
                y_set=train_eye_centre_df[["lx", "ly", "rx", "ry"]],
                root_directory=train_img_folder,
                img_dim=img_height,
                augmentation=None,
                shuffle=True
            )

            val_ds = CustomDataGenerator(
                batch_size=32,
                x_set=val_eye_centre_df[["filename"]],
                y_set=val_eye_centre_df[["lx", "ly", "rx", "ry"]],
                root_directory=train_img_folder,
                img_dim=img_height,
                augmentation=None,
                shuffle=True
            )
            model_directory = os.path.join(root_folder, "models", config.model_directory_mobilenet_2) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            
            model, keyword = load_mobilenetv2_model(ending_layers=2) # edit this for different models
        
        elif test == 9:
            model_directory = os.path.join(root_folder, "models", config.model_directory_inception_0) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_inception_model(ending_layers=0)
        
        elif test == 10:
            model_directory = os.path.join(root_folder, "models", config.model_directory_inception_1) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_inception_model(ending_layers=1)

        elif test == 11:
            model_directory = os.path.join(root_folder, "models", config.model_directory_inception_2) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_inception_model(ending_layers=2)

        elif test == 12:
            model_directory = os.path.join(root_folder, "models", config.model_directory_vgg_0) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_vgg_model(ending_layers=0)

        elif test == 13:
            model_directory = os.path.join(root_folder, "models", config.model_directory_vgg_1) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_vgg_model(ending_layers=1)

        elif test == 14:
            model_directory = os.path.join(root_folder, "models", config.model_directory_vgg_2) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_vgg_model(ending_layers=2)

        elif test == 15:
            model_directory = os.path.join(root_folder, "models", config.model_directory_resnet_0) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_resnet152v2_model(ending_layers=0)

        elif test == 16:
            model_directory = os.path.join(root_folder, "models", config.model_directory_resnet_1) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_resnet152v2_model(ending_layers=1)

        elif test == 17:
            model_directory = os.path.join(root_folder, "models", config.model_directory_resnet_2) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_resnet152v2_model(ending_layers=2)

        elif test == 18:
            model_directory = os.path.join(root_folder, "models", config.model_directory_efficient_0) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_efficientnetb6_model(ending_layers=0)

        elif test == 19:
            model_directory = os.path.join(root_folder, "models", config.model_directory_efficient_1) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_efficientnetb6_model(ending_layers=1)

        elif test == 20:
            model_directory = os.path.join(root_folder, "models", config.model_directory_efficient_2) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_efficientnetb6_model(ending_layers=2)

        elif test == 21:
            model_directory = os.path.join(root_folder, "models", config.model_directory_convnet_0) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_convnext_model(ending_layers=0)

        elif test == 22:
            model_directory = os.path.join(root_folder, "models", config.model_directory_convnet_1) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_convnext_model(ending_layers=1)

        elif test == 23:
            model_directory = os.path.join(root_folder, "models", config.model_directory_convnet_2) ## update to relevant directory
            weights_filepath = os.path.join(model_directory, config.best_model) ## update to relevant hdf5 file
            model, keyword = load_convnext_model(ending_layers=2)

        # get model architecture
        model.load_weights(weights_filepath)

        # to get around bug in keras/tf that prevents saving and loading weights from models with a combination
        # of frozen and unfrozen layers
        for layer in model.layers:
            layer.trainable = True

        # compile model with key parameters defined below
        opt = Adam(lr = 1e-4)
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=opt, metrics=[tf.keras.metrics.MeanAbsoluteError()])

        # train model
        # Callbacks - recording the intermediate training results which can be visualised on tensorboard
        subFolderLog = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboardPath = os.path.join(os.getcwd(), "models", subFolderLog + "_" + keyword)
        checkpointPath = os.path.join(os.getcwd(), "models" , subFolderLog + "_" + keyword)
        # checkpointPath = checkpointPath + "/" + keyword + "-weights-improve-{epoch:02d}-{val_loss:02f}.hdf5"
        checkpointPath = checkpointPath + "/" + "best_model.hdf5"

        # reduce learning rate if loss stagnates
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor = 0.5, patience=10, min_lr = 0.000001)

        callbacks = [
            TensorBoard(log_dir=tensorboardPath, histogram_freq=0, write_graph=True, write_images=True),
            ModelCheckpoint(filepath=checkpointPath, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
            reduce_lr
            # EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        ]

        print("[INFO] Previously trained model loaded and compiled...")

        num_epochs = 250

        print(model.summary())

        # retrain the model
        print("[INFO] Retraining eye centre localisation model...")
        H = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=num_epochs,
            callbacks=callbacks,
            verbose=1)
        
        # calculate results
        predictions = np.array([])
        labels =  np.array([])
        count=0 
        
        for x, y in val_ds:

            # print(y)

            if len(predictions.shape) == 1:
                predictions = model.predict(x)
                labels = y
                
            else: 
                predictions = np.concatenate([predictions,  model.predict(x)])
                labels = np.concatenate([labels, y])

            # print(predictions, labels)

        predictions = (predictions*(960))
        labels = (labels*960)

        calculate_visual_angle_error(predictions, labels)

        # calculate Euclidean distance between predictions and ground truth
        euclidean_matix_left_eye = ((predictions[:,0]-labels[:,0])**2+(predictions[:,1]-labels[:,1])**2)**0.5
        euclidean_matrix_right_eye = ((predictions[:,2]-labels[:,2])**2+(predictions[:,3]-labels[:,3])**2)**0.5
        euclidean_matrix = np.column_stack((euclidean_matix_left_eye, euclidean_matrix_right_eye))
        euclidean_matrix_max = np.amax(euclidean_matrix, axis=1)

        # calculate ipd
        ipd_matrix = ((labels[:,3]-labels[:,1])**2+(labels[:,2]-labels[:,0])**2)**0.5

        # normalise error by ipd
        e_max = euclidean_matrix_max/ipd_matrix
        e_left = euclidean_matix_left_eye/ipd_matrix
        e_right = euclidean_matrix_right_eye/ipd_matrix

        # calculate predictions above required vals
        vvs_error, vs_error, half_pupil_all_error, full_pupil_all_error, large_error_all_error, v_large_error_all_error = [], [], [], [], [], []
        for error in [e_max, e_left, e_right]:

            error_vvs_pupil = (error <= 0.005).sum()
            error_vs_pupil = (error <= 0.01).sum()
            error_half_pupil = (error <= 0.025).sum()
            error_full_pupil = (error <= 0.05).sum()
            error_large_error = (error <= 0.1).sum()
            error_v_large_error = (error <= 0.25).sum()

            vvs_error.append(error_vvs_pupil/e_max.shape[0])
            vs_error.append(error_vs_pupil/e_max.shape[0])
            half_pupil_all_error.append(error_half_pupil/e_max.shape[0])
            full_pupil_all_error.append(error_full_pupil/e_max.shape[0])
            large_error_all_error.append(error_large_error/e_max.shape[0])
            v_large_error_all_error.append(error_v_large_error/e_max.shape[0])

        print("Format: [max error, left eye error, right eye error]")
        print("Total predictions: ", str(e_max.shape))
        error_label = ["VVS error", "VS error", "Half pupil", "Full pupil", "Large error", "V large error"]
        count = 0
        for error in [vvs_error, vs_error, half_pupil_all_error, full_pupil_all_error, large_error_all_error, v_large_error_all_error]:
            print(error_label[count])
            print(error)

            count += 1

        accuracy_df.append([test, vvs_error, vs_error, half_pupil_all_error, full_pupil_all_error, large_error_all_error, v_large_error_all_error])
        accuracy_df_to_save = pd.DataFrame(data=accuracy_df)
        print(accuracy_df)
        print(accuracy_df_to_save)
        accuracy_df_to_save.to_csv("validation_results_1st_train.csv")