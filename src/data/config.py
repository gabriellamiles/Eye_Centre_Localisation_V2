import os

eye_centre_folders_raw = [os.path.join(os.getcwd(), "data", "raw", "centres"),
                          os.path.join(os.getcwd(), "data", "raw", "labels_eme2")]

eye_centre_columns = ["filename", "lx", "ly", "rx", "ry"]
bounding_columns = ["filename", "LE_left", "LE_top", "LE_right", "LE_bottom", "RE_left", "RE_top", "RE_right", "RE_bottom"]

bounding_box_folder = os.path.join(os.getcwd(), "data", "raw", "bounding_boxes")
combined_eye_centre_folder = os.path.join(os.getcwd(), "data", "raw", "combined_centres")
final_data_folder = os.path.join(os.getcwd(), "data", "processed", "combined_labels")