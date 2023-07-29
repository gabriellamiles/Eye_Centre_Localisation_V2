import config
import model_utils
import models
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from eye_data import Dataset
from data_generator import CustomDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier


def load_dataset():

    # load labels
    labels = model_utils.load_labels() # make sure label_folder is defined as correct location (in config file)
    labels = labels.sample(frac=1, random_state=42) # shuffle labels

    # initialise dataset
    dataset = Dataset(labels=labels)

    # initial preprocessing here
    dataset.remove_blinks()
    dataset.get_train_val_test(dataset.eye_centres_no_blinks) # split into train/test data
    dataset.get_k_folds() # apply k fold cross validation to training sets (folds = 5)

    return dataset

def load_data_generators(img_dim, train_filenames, train_labels, val_filenames, val_labels, val_batch_size=1):
    # initialise data generators
    eye_data_generator_train = CustomDataGenerator(
            batch_size=batch_size, 
            x_set=train_filenames, 
            y_set=train_labels, 
            root_directory=test_filepath,
            img_dim=img_dim, 
            augmentation="nothing",
            shuffle=True
            )
    
    eye_data_generator_val = CustomDataGenerator(
            batch_size=val_batch_size, 
            x_set=val_filenames, 
            y_set=val_labels, 
            root_directory=test_filepath,
            img_dim=img_dim, 
            augmentation="nothing",
            shuffle=False
            )
    return eye_data_generator_train, eye_data_generator_val

def examine_batches(generator, dim):

    images = next(iter(generator))
    nrows, ncols = 4, 2
    
    fig = plt.figure(figsize=(10,10))
    for i in range(8):

        ax = fig.add_subplot(nrows, ncols, i+1)
        plt.imshow(images[0][i].astype('uint8'))
        circ1 = Circle((images[1][i,0]*dim, images[1][i,1]*dim),5)
        circ2 = Circle((images[1][i,2]*dim, images[1][i,3]*dim),5)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        plt.axis(False)

    plt.savefig("test.png")
    plt.show()

def get_fold_data(dataset, k_fold):

    # get relevant training and validation folds  
    train_labels, val_labels = dataset.train_labels.iloc[dataset.kf_train_indices[k_fold]], dataset.train_labels.iloc[dataset.kf_val_indices[k_fold]]
    train_filenames, train_labels = train_labels[["filename"]], train_labels[config.eye_centre_cols]
    val_filenames, val_labels = val_labels[["filename"]], val_labels[config.eye_centre_cols]

    return train_filenames, train_labels, val_filenames, val_labels

if __name__ == '__main__':

    # initialise key parameters
    batch_size = 16
    retrain_existing_model = 0
    test_filepath = os.path.join(os.getcwd(), "data", "processed", "mnt", "eme2_square_imgs")
    
    # load dataset
    dataset = load_dataset()
    
    for k_fold in range(0, 5):

        train_filenames, train_labels, val_filenames, val_labels = get_fold_data(dataset, k_fold)
        
        # load model to train
        # model_dim = 299
        # get data generators
        # eye_data_generator_train, eye_data_generator_val = load_data_generators(model_dim, train_filenames, train_labels, val_filenames, val_labels)
        # examine a batch of 8 images
        # examine_batches(eye_data_generator_train, model_dim)

    #     # load and train mdoel
    #     inception_estimator = models.Inception_model((model_dim, model_dim, 3), test_num=k_fold, output_parameters=4, batch_size=batch_size, directory="models/updated_models")
    #     inception_estimator.train_model(eye_data_generator_train, eye_data_generator_val)
    #     inception_estimator.predict_model(eye_data_generator_val)
    #     inception_estimator.save_results(val_labels, val_filenames)
    #     # load and train mdoel
    #     xception_estimator = models.Xception_model((model_dim, model_dim, 3), test_num=k_fold, output_parameters=4, batch_size=batch_size, directory="models/updated_models")
    #     xception_estimator.train_model(eye_data_generator_train, eye_data_generator_val)
    #     xception_estimator.predict_model(eye_data_generator_val)
    #     xception_estimator.save_results(val_labels, val_filenames)

    #     # load model to train
        model_dim = 224
    #     # get data generators
        eye_data_generator_train, eye_data_generator_val = load_data_generators(model_dim, train_filenames, train_labels, val_filenames, val_labels)

    #     vgg_estimator = models.Inception_model((model_dim, model_dim, 3), test_num=k_fold, output_parameters=4, batch_size=batch_size, directory="models/updated_models")
    #     vgg_estimator.train_model(eye_data_generator_train, eye_data_generator_val)
    #     vgg_estimator.predict_model(eye_data_generator_val)
    #     vgg_estimator.save_results(val_labels, val_filenames)

        resnet50_estimator = models.ResNet50_model((model_dim, model_dim, 3), test_num=k_fold, output_parameters=4, batch_size=batch_size, directory="models/updated_models")
        resnet50_estimator.train_model(eye_data_generator_train, eye_data_generator_val)
        resnet50_estimator.predict_model(eye_data_generator_val)
        resnet50_estimator.save_results(val_labels, val_filenames)

        
