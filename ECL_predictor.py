import os
import cv2
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from build_dataset import build_dataset, partition_dataset
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

def load_images_to_test(rootFolder):

    images_to_label = os.path.join(rootFolder, "test.csv")
    df = pd.read_csv(images_to_label)[["filename"]]

    return df

if __name__ == '__main__':

    # retrieve bounding box and eye centre coordinates
    bdbx_labels = os.path.join(config.ROOTFOLDER.replace("improved_ECL", "eye_region_detector"), "output", "eye_region", config.PARTICIPANT_CSV_FILE)
    EC_filePath = os.path.join(config.ROOTFOLDER.replace("improved_ECL", "eye_centre_localisation"), "labels_eme2", config.PARTICIPANT_CSV_FILE)

    # BUILD DATASET
    centres_df = pd.read_csv(EC_filePath)[['filename', 'lx', 'ly', 'rx', 'ry']]
    region_df = pd.read_csv(bdbx_labels)[['filename', 'startLX', 'startLY', 'endLX', 'endLY', 'startRX', 'startRY', 'endRX', 'endRY']]
    data, targets, filenames = build_dataset(config.IMGFOLDER, centres_df, region_df, config.VGG_INPUT_SHAPE)

    # build train, validation, and test split using left and right eyes
    trainImages, testImages, trainTargets, testTargets, trainFilenames, testFilenames = partition_dataset(data, targets, filenames)
    print(testTargets)

    # LOAD TRAINED MODEL TO MAKE PREDICTIONS
    print("[INFO] Using saved model to make predictions...")
    model = load_model(os.path.join(os.getcwd(), "output", "vgg_detector.h5"))

    euclidean_error = []
    x_error = []
    y_error = []

    count = 0
    for image in testImages:

        original_image = np.copy(image)
        image = np.expand_dims(image,axis=0)

        # make eye centre predictions on the input image
        preds = model.predict(image)[0]
        (x, y) = preds

        x = int(x*config.VGG_INPUT_SHAPE)
        y = int(y*config.VGG_INPUT_SHAPE)

        gth_x = int(testTargets[count][0]*config.VGG_INPUT_SHAPE)
        gth_y = int(testTargets[count][1]*config.VGG_INPUT_SHAPE)
        # print(x, y, gth_x, gth_y)

        dist = ((gth_x-x)**2 + (gth_y-y)**2)**0.5
        euclidean_error.append(int(dist))
        x_error.append(gth_x-x)
        y_error.append(gth_y-y)

        original_image = original_image*255
        # print(type(original_image))

        cv_img = original_image.astype(np.uint8)

        cv2.circle(cv_img, (x, y), 4, (0, 255, 0), -1)
        cv2.circle(cv_img, (gth_x, gth_y), 4, (255, 0, 0), -1)
        # cv2.imshow("image", cv_img)
        # cv2.waitKey(0)

        count += 1

    # plot histogram of errors
    sequence = [i for i in range(int(min(euclidean_error)), 30)]
    print(max(euclidean_error))
    n, bins, patches = plt.hist(x=euclidean_error, bins=sequence, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Count')
    plt.title('Total Pixel Error in Predictions for Eye Centre')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(os.path.join(os.getcwd(), "output", "eye_centre_error.png"), dpi='figure', format="png")
