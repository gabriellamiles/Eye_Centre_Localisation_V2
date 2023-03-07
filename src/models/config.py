import os

# key filepaths
root_folder = os.getcwd()
square_img_folder = os.path.join(root_folder, "data", "processed", "mnt", "eme2_square_imgs")
cropped_img_folder = os.path.join(root_folder, "data", "processed", "mnt", "cropped_imgs")
label_folder = os.path.join(root_folder, "data", "processed", "combined_labels")
test_split_save_location = os.path.join(root_folder, "data", "processed", "test_split.csv")

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
