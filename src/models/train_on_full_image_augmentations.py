import os 
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import config
import model_utils
import models
from eye_data import Dataset
from data_generator import CustomDataGenerator

if __name__ == '__main__':

    print("hello")
    test_parameters = model_utils.test_configuration(os.path.join(os.getcwd(), "src", "models", "augmentation_tests.csv"))
    model_input_dim = (test_parameters["input_dim"], test_parameters["input_dim"], 3)
    test_num = test_parameters["test_num"]
    batch_size=int(test_parameters["batch_size"])
    augmentation = test_parameters["augmentation"]

    print("****************************************")
    print(test_parameters)

    # load labels
    labels = model_utils.load_labels() # make sure label_folder is defined as correct location (in config file)
    labels = labels.sample(frac=1, random_state=42) # shuffle labels
    
    # initialise dataset
    dataset = Dataset(labels=labels)
    # initial preprocessing here
    dataset.remove_blinks()
    dataset.get_train_val_test(dataset.eye_centres_no_blinks) # split into train/test data
    dataset.get_k_folds() # apply k fold cross validation to training sets (folds = 5)

    # get relevant training and validation folds  
    train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[test_parameters["fold"]]], dataset.train_labels.iloc[dataset.kf_val_indices[test_parameters["fold"]]]
    train_filenames, train_labels = train_labels[["filename"]], train_labels[config.eye_centre_cols]
    val_filenames, val_labels = val_labels[["filename"]], val_labels[config.eye_centre_cols]

    # load corresponding images at correct size for model
    test_filepath = os.path.join(os.getcwd(), "data", "processed", "mnt", "eme2_square_imgs")
    train_images, val_images = [], []

    # if augmentation is required use custom data generators
    eye_data_generator_train, eye_data_generator_val = None, None
    if "None" not in augmentation:
        eye_data_generator_train = CustomDataGenerator(
            batch_size=batch_size, 
            x_set=train_filenames, 
            y_set=train_labels, 
            root_directory=test_filepath,
            img_dim=test_parameters["input_dim"], 
            augmentation=test_parameters["augmentation"],
            shuffle=True
            )
        
        eye_data_generator_val = CustomDataGenerator(
            batch_size=batch_size, 
            x_set=val_filenames, 
            y_set=val_labels, 
            root_directory=test_filepath,
            img_dim=test_parameters["input_dim"], 
            augmentation="nothing",
            shuffle=False
            )
    else:
        # load images and labels into memory like normal
        train_labels, val_labels = train_labels/960, val_labels/960

        print("Loading training images...")
        for row in range(train_filenames.shape[0]):
            img_filepath = os.path.join(test_filepath, train_filenames.iloc[row, 0] )
            image = tf.keras.preprocessing.image.load_img(img_filepath, color_mode="rgb", target_size=(test_parameters["input_dim"], test_parameters["input_dim"]))
            image = tf.keras.preprocessing.image.img_to_array(image).astype(int)
            train_images.append(image)


        print("Loading validation images...")
        for row in range(val_filenames.shape[0]):
            img_filepath = os.path.join(test_filepath, val_filenames.iloc[row, 0] )
            image = tf.keras.preprocessing.image.load_img(img_filepath, color_mode="rgb", target_size=(test_parameters["input_dim"], test_parameters["input_dim"]))
            image = tf.keras.preprocessing.image.img_to_array(image).astype(int)
            val_images.append(image)

        train_images, val_images = np.array(train_images), np.array(val_images)
        train_labels, val_labels = np.array(train_labels), np.array(val_labels)

        print(train_images.shape, val_images.shape, train_labels.shape, val_labels.shape)

    # examine a batch of images
    # images = next(iter(eye_data_generator_train))
    # nrows, ncols = 4, 2
    
    # fig = plt.figure(figsize=(10,10))
    # for i in range(8):

    #     ax = fig.add_subplot(nrows, ncols, i+1)
    #     plt.imshow(images[0][i].astype('uint8'))
    #     circ1 = Circle((images[1][i,0]*test_parameters["input_dim"], images[1][i,1]*test_parameters["input_dim"]),5)
    #     circ2 = Circle((images[1][i,2]*test_parameters["input_dim"], images[1][i,3]*test_parameters["input_dim"]),5)
    #     ax.add_patch(circ1)
    #     ax.add_patch(circ2)
    #     plt.axis(False)

    # plt.savefig("test.png")
    # plt.show()

    # load and train relevant model
    if test_parameters["model"] == "vgg":
        # load model to train
        vgg = models.VGG_model(model_input_dim, test_num=test_num, output_parameters=4, batch_size=batch_size)
        if "None" not in augmentation:
            vgg.train_model(eye_data_generator_train, eye_data_generator_val)
        else:
            # train model with data loaded into memory 
            vgg.train_model_memory(train_images, train_labels, val_images, val_labels)

        vgg.plot_loss_curves()
        
    elif test_parameters["model"] == "resnet50":    
        # resnet model
        resnet_50 = models.ResNet50_model(model_input_dim, test_num=test_num, output_parameters=4, batch_size=batch_size)
        
        if "None" not in augmentation:
            resnet_50.train_model(eye_data_generator_train, eye_data_generator_val)
        else:
            # train model with data loaded into memory 
            resnet_50.train_model_memory(train_images, train_labels, val_images, val_labels)

        resnet_50.plot_loss_curves()

    elif test_parameters["model"] == "inception":
        # inception model
        inception = models.Inception_model(model_input_dim, test_num=test_num, output_parameters=4, batch_size=batch_size)

        if "None" not in augmentation:
            inception.train_model(eye_data_generator_train, eye_data_generator_val)
        else:
            # train model with data loaded into memory 
            inception.train_model_memory(train_images, train_labels, val_images, val_labels)

        inception.plot_loss_curves()

    elif test_parameters["model"] == "inception_resnet_v2":
        # inception with residual connections
        inception_resnet_v2 = models.InceptionResNetV2_model(model_input_dim, test_num=test_num, output_parameters=4, batch_size=batch_size)
        if "None" not in augmentation:
            inception_resnet_v2.train_model(eye_data_generator_train, eye_data_generator_val)
        else:
            # train model with data loaded into memory 
            inception_resnet_v2.train_model_memory(train_images, train_labels, val_images, val_labels)
        inception_resnet_v2.plot_loss_curves()

    elif test_parameters["model"] == "xception":
        # xception model
        xception = models.Xception_model(model_input_dim, test_num=test_num, output_parameters=4, batch_size=batch_size)
        if "None" not in augmentation:
            xception.train_model(eye_data_generator_train, eye_data_generator_val)
        else:
            # train model with data loaded into memory 
            xception.train_model_memory(train_images, train_labels, val_images, val_labels)
        xception.plot_loss_curves()        