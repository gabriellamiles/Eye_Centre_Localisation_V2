import os

# key filepaths
root_folder = os.getcwd()
square_img_folder = os.path.join(root_folder, "data", "processed", "mnt", "eme2_square_imgs")
cropped_img_folder = os.path.join(root_folder, "data", "processed", "mnt", "eye_patches")
label_folder = os.path.join(root_folder, "data", "processed", "combined_labels")
full_bb_ec_labels = os.path.join(root_folder, "data", "raw", "final_dataset")


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
