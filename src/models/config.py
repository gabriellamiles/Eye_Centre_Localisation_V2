import os

# current model

# hyperparameters
INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32
VGG_INPUT_SHAPE = 224
INCEPTIONV3_INPUT_SHAPE = 299

# data augmentation information
HORIZONTAL_FLIPS = 0.5 # 50% chance

# common filepaths
ROOTFOLDER = os.getcwd()
IMGFOLDER = os.path.join(ROOTFOLDER.replace("improved_ECL", "eye_centre_localisation"), "mnt", "eme2_square_imgs")

# misc
PARTICIPANT_CSV_FILE = "029_3.csv"
