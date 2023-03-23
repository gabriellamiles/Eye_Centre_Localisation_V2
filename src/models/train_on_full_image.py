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

    test_parameters = model_utils.test_configuration(os.path.join(os.getcwd(), "src", "models", "model_training_runs.csv"))
    model_input_dim = (test_parameters["input_dim"], test_parameters["input_dim"], 3)
    test_num = test_parameters["test_num"]
    batch_size=int(test_parameters["batch_size"])

    # load labels
    labels = model_utils.load_labels() # make sure label_folder is defined as correct location (in config file)
    labels = labels.sample(frac=1, random_state=42) # shuffle labels
    
    # initialise dataset
    dataset = Dataset(labels=labels)
    dataset.get_train_val_test() # split into train/test data
    dataset.get_k_folds() # apply k fold cross validation to training sets (folds = 5)

    val_size=2472
    if batch_size==8:
        val_size = 2472
    elif batch_size>8:
        val_size=2472-(batch_size+8)

    # get relevant training and validation folds  
    train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[test_parameters["fold"]]], dataset.train_labels.iloc[dataset.kf_val_indices[test_parameters["fold"]]]
    train_filenames, train_labels = train_labels[["filename"]], train_labels[config.eye_centre_cols]
    val_filenames, val_labels = val_labels[["filename"]].iloc[:val_size, :], val_labels[config.eye_centre_cols].iloc[:2472, :]

    # load corresponding images at correct size for model
    test_filepath = os.path.join(os.getcwd(), "data", "processed", "mnt", "eme2_square_imgs")
    
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

    # examine a batch of images
    images = next(iter(eye_data_generator_train))
    nrows, ncols = 4, 2
    
    fig = plt.figure(figsize=(10,10))
    for i in range(8):

        ax = fig.add_subplot(nrows, ncols, i+1)
        plt.imshow(images[0][i].astype('uint8'))
        circ1 = Circle((images[1][i,0]*test_parameters["input_dim"], images[1][i,1]*test_parameters["input_dim"]),5)
        circ2 = Circle((images[1][i,2]*test_parameters["input_dim"], images[1][i,3]*test_parameters["input_dim"]),5)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        plt.axis(False)

    plt.savefig("test.png")
    plt.show()

    # load and train relevant model
    if test_parameters["model"] == "vgg":
        # load model to train
        vgg = models.VGG_model(model_input_dim, test_num=test_num, output_parameters=4, batch_size=batch_size)
        vgg.train_model(eye_data_generator_train, eye_data_generator_val)
        vgg.plot_loss_curves()

        # make accuracy predictions on validation set
        y_preds = vgg.predict_model(eye_data_generator_val)
        val_labels_test = np.array(val_labels)
        print(y_preds.shape, val_labels_test.shape)

        # calculate accuracy
        diff = val_labels_test-y_preds
        squared = np.square(diff)

        left_eye = np.sum(squared[:, 0:2], axis=1)
        right_eye = np.sum(squared[:,2:4], axis=1)

        left_eye_accuracy = np.sqrt(left_eye)
        left_eye_acc = np.sum(left_eye_accuracy)/y_preds.shape[0]

        right_eye_accuracy = np.sqrt(right_eye)
        right_eye_acc = np.sum(right_eye_accuracy)/y_preds.shape[0]

        d = {'left eye accuracy': [left_eye_acc], 'right eye accuracy': [right_eye_acc]}
        data = pd.DataFrame(d)
        data.to_csv(str(test_num)+".csv")

        print("***************ACCURACY:")
        print(test_parameters)
        print(left_eye_acc, right_eye_acc)
        
    elif test_parameters["model"] == "resnet50":    
        # resnet model
        resnet_50 = models.ResNet50_model(model_input_dim, test_num=test_num, output_parameters=4, batch_size=batch_size)
        resnet_50.train_model(eye_data_generator_train, eye_data_generator_val)
        resnet_50.plot_loss_curves()

        # make accuracy predictions on validation set
        y_preds = resnet_50.predict_model(eye_data_generator_val)
        val_labels_test = np.array(val_labels)
        print(y_preds.shape, val_labels_test.shape)

        # calculate accuracy
        diff = val_labels_test-y_preds
        squared = np.square(diff)

        left_eye = np.sum(squared[:, 0:2], axis=1)
        right_eye = np.sum(squared[:,2:4], axis=1)

        left_eye_accuracy = np.sqrt(left_eye)
        left_eye_acc = np.sum(left_eye_accuracy)/y_preds.shape[0]

        right_eye_accuracy = np.sqrt(right_eye)
        right_eye_acc = np.sum(right_eye_accuracy)/y_preds.shape[0]

        d = {'left eye accuracy': [left_eye_acc], 'right eye accuracy': [left_eye_acc]}
        data = pd.DataFrame(d)
        data.to_csv(str(test_num)+".csv")

        print("***************ACCURACY:")
        print(test_parameters)
        print(left_eye_acc, right_eye_acc)

    elif test_parameters["model"] == "inception":
        # inception model
        inception = models.Inception_model(model_input_dim, test_num=test_num, output_parameters=4, batch_size=batch_size)
        inception.train_model(eye_data_generator_train, eye_data_generator_val)
        inception.plot_loss_curves()

        # make accuracy predictions on validation set
        y_preds = inception.predict_model(eye_data_generator_val)
        val_labels_test = np.array(val_labels)
        print(y_preds.shape, val_labels_test.shape)

        # calculate accuracy
        diff = val_labels_test-y_preds
        squared = np.square(diff)

        left_eye = np.sum(squared[:, 0:2], axis=1)
        right_eye = np.sum(squared[:,2:4], axis=1)

        left_eye_accuracy = np.sqrt(left_eye)
        left_eye_acc = np.sum(left_eye_accuracy)/y_preds.shape[0]

        right_eye_accuracy = np.sqrt(right_eye)
        right_eye_acc = np.sum(right_eye_accuracy)/y_preds.shape[0]

        d = {'left eye accuracy': [left_eye_acc], 'right eye accuracy': [left_eye_acc]}
        data = pd.DataFrame(d)
        data.to_csv(str(test_num)+".csv")

        print("***************ACCURACY:")
        print(test_parameters)
        print(left_eye_acc, right_eye_acc)

    elif test_parameters["model"] == "inception_resnet_v2":
        # inception with residual connections
        inception_resnet_v2 = models.InceptionResNetV2_model(model_input_dim, test_num=test_num, output_parameters=4, batch_size=batch_size)
        inception_resnet_v2.train_model(eye_data_generator_train, eye_data_generator_val)
        inception_resnet_v2.plot_loss_curves()

        # make accuracy predictions on validation set
        y_preds = inception_resnet_v2.predict_model(eye_data_generator_val)
        val_labels_test = np.array(val_labels)
        print(y_preds.shape, val_labels_test.shape)

        # calculate accuracy
        diff = val_labels_test-y_preds
        squared = np.square(diff)

        left_eye = np.sum(squared[:, 0:2], axis=1)
        right_eye = np.sum(squared[:,2:4], axis=1)

        left_eye_accuracy = np.sqrt(left_eye)
        left_eye_acc = np.sum(left_eye_accuracy)/y_preds.shape[0]

        right_eye_accuracy = np.sqrt(right_eye)
        right_eye_acc = np.sum(right_eye_accuracy)/y_preds.shape[0]

        d = {'left eye accuracy': [left_eye_acc], 'right eye accuracy': [left_eye_acc]}
        data = pd.DataFrame(d)
        data.to_csv(str(test_num)+".csv")

        print("***************ACCURACY:")
        print(test_parameters)
        print(left_eye_acc, right_eye_acc)

    elif test_parameters["model"] == "xception":
        # xception model
        xception = models.Xception_model(model_input_dim, test_num=test_num, output_parameters=4, batch_size=batch_size)
        xception.train_model(eye_data_generator_train, eye_data_generator_val)
        xception.plot_loss_curves()

        # make accuracy predictions on validation set
        y_preds = xception.predict_model(eye_data_generator_val)
        val_labels_test = np.array(val_labels)
        print(y_preds.shape, val_labels_test.shape)

        # calculate accuracy
        diff = val_labels_test-y_preds
        squared = np.square(diff)

        left_eye = np.sum(squared[:, 0:2], axis=1)
        right_eye = np.sum(squared[:,2:4], axis=1)

        left_eye_accuracy = np.sqrt(left_eye)
        left_eye_acc = np.sum(left_eye_accuracy)/y_preds.shape[0]

        right_eye_accuracy = np.sqrt(right_eye)
        right_eye_acc = np.sum(right_eye_accuracy)/y_preds.shape[0]

        d = {'left eye accuracy': [left_eye_acc], 'right eye accuracy': [left_eye_acc]}
        data = pd.DataFrame(d)
        data.to_csv(str(test_num)+".csv")

        print("***************ACCURACY:")
        print(test_parameters)
        print(left_eye_acc, right_eye_acc)

        