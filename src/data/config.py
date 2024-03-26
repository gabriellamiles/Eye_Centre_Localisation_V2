import os

eye_centre_folders_raw = [os.path.join(os.getcwd(), "data", "raw", "centres"),
                          os.path.join(os.getcwd(), "data", "raw", "labels_eme2")]

eye_centre_columns = ["filename", "lx", "ly", "rx", "ry"]
bounding_columns = ["filename", "relative_LE_left", "relative_LE_top", "relative_LE_right", "relative_LE_bottom", "relative_RE_left", "relative_RE_top", "relative_RE_right", "relative_RE_bottom"]

bounding_box_folder = os.path.join(os.getcwd(), "data", "raw", "bounding_boxes")
combined_eye_centre_folder = os.path.join(os.getcwd(), "data", "raw", "combined_centres")
final_data_folder = os.path.join(os.getcwd(), "data", "raw", "final_dataset")