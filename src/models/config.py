import os

# key filepaths
root_folder = os.getcwd()
square_img_folder = os.path.join(root_folder, "data", "processed", "mnt1", "eme2_square_imgs")
cropped_img_folder = os.path.join(root_folder, "data", "processed", "mnt1", "eye_patches")
label_folder = os.path.join(root_folder, "data", "processed", "combined_labels")
full_bb_ec_labels = os.path.join(root_folder, "data", "raw", "final_dataset_recent")


test_split_save_location = os.path.join(root_folder, "data", "processed", "test_split.csv")
train_split_save_location = os.path.join(root_folder, "data", "processed", "train_split.csv")
model_save_folder = os.path.join(root_folder, "models", "single_eye")

# columns to return different data when initialising datasets
eye_centre_cols = ["lx", "ly", "rx", "ry"]
filename = ["filename"]

# hyperparameters
INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32
VGG_INPUT_SHAPE = 224
INCEPTIONV3_INPUT_SHAPE = 299

# data augmentation information
HORIZONTAL_FLIPS = 0.5 # 50% chance

# misc
PARTICIPANT_CSV_FILE = "029_3.csv"

# inputs
model_name = "test_0_inception_20230701_143553"


## retrain filepaths
best_model = "best_model.hdf5"
model_directory_inception_resnet_0 = "20240306_130614_inceptionresnetv2_ending0"
weights_file_inception_resnet_0 = "inceptionresnetv2_ending0-weights-improve-09-0.038972.hdf5"
model_directory_inception_resnet_1 = "20240306_224741_inceptionresnetv2_ending1"
weights_file_inception_resnet_1 = "inceptionresnetv2_ending1-weights-improve-25-0.025500.hdf5"
model_directory_inception_resnet_2 = "20240307_083040_inceptionresnetv2_ending2"
weights_file_inception_resnet_2 = "inceptionresnetv2_ending2-weights-improve-21-0.020301.hdf5"

model_directory_xception_0 = "20240307_181731_xception_ending0"
weights_file_xception_0 = "xception_ending0-weights-improve-35-0.011797.hdf5"
model_directory_xception_1 = "20240308_034819_xception_ending1"
weights_file_xception_1 = "xception_ending1-weights-improve-80-0.014474.hdf5"
model_directory_xception_2 = "20240308_131928_xception_ending2"
weights_file_xception_2 = "xception_ending2-weights-improve-62-0.010961.hdf5"

model_directory_mobilenet_0 = "20240309_063921_mobilenetv2_ending0"
weights_file_mobilenet_0 = "mobilenetv2_ending0-weights-improve-29-0.012566.hdf5"
model_directory_mobilenet_1 = "20240308_225638_mobilenetv2_ending1"
weights_file_mobilenet_1 = "mobilenetv2_ending1-weights-improve-80-0.031817.hdf5"
model_directory_mobilenet_2 = "20240309_142223_mobilenetv2_ending2"
weights_file_mobilenet_2 = "mobilenetv2_ending2-weights-improve-63-0.015553.hdf5"

model_directory_inception_0 = "20240310_155741_inceptionv3_ending0"
model_directory_inception_1 = "20240311_011520_inceptionv3_ending1"
model_directory_inception_2 = "20240311_103329_inceptionv3_ending2"

model_directory_vgg_0 = "20240311_195607_vgg16_ending0"
model_directory_vgg_1 = "20240312_045851_vgg16_ending1"
model_directory_vgg_2 = "20240312_141903_vgg16_ending2"

model_directory_resnet_0 = "20240312_232943_resnetv2_ending0"
model_directory_resnet_1 = "20240313_113441_resnetv2_ending1"
model_directory_resnet_2 = "20240313_234054_resnetv2_ending2"

model_directory_efficient_0 = "20240317_225021_efficientnetb6_ending0"
model_directory_efficient_1 = "20240318_150009_efficientnetb6_ending1"
model_directory_efficient_2 = "20240319_062947_efficientnetb6_ending2"

model_directory_convnet_0 = "20240319_220440_convnext_ending0"
model_directory_convnet_1 = "20240320_120355_convnext_ending1"
model_directory_convnet_2 = "20240321_020903_convnext_ending2"

# test filepaths
model_directory_test = "20240306_121913_inceptionresnetv2_ending0"
weights_file_test = "best_model.hdf5"

# hand built cnn filepaths
hb_cnn_directory = "20240117_015233_hand_built_cnn_17"
hb_cnn_weights = "hand_built_cnn_17-weights-improve-200-0.002500.hdf5"
