import os
import cv2

# Set the input and output directories


input_directory = os.path.join(os.getcwd(), "data", "convert_to_square")#, "013", "0")
output_directory = os.path.join(os.getcwd(), "data", "squared_imgs")#, "013", "0")

for folder in os.listdir(input_directory):

    participant_folder = os.path.join(input_directory, folder)

    for trial in ["0", "1", "2", "3"]:

        trial_folder = os.path.join(participant_folder, trial)
        output_folder = os.path.join(output_directory, folder, trial)

        if not os.path.exists(trial_folder):
            continue

        # Ensure the output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            print(output_folder + " already squared. ")
            continue

        # Set the target square size
        target_size = 960

        # List all image files in the input directory
        image_files = [f for f in os.listdir(trial_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        # Process each image
        for image_file in image_files:

            input_path = os.path.join(trial_folder, image_file)
            output_path = os.path.join(output_folder, image_file)

            # Read the image
            image = cv2.imread(input_path)

            # Get the dimensions of the image
            height, width, channels = image.shape

            # Calculate the size of the black border to make the image square
            border_size = abs(width - height) // 2

            # Create a black border around the image
            if width > height:
                border = cv2.copyMakeBorder(image, border_size, border_size, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                border = cv2.copyMakeBorder(image, 0, 0, border_size, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Resize the image to the target size (960x960)
            squared_image = cv2.resize(border, (target_size, target_size))

            # Save the squared image to the output directory
            cv2.imwrite(output_path, squared_image)

        print("Conversion completed. Square images saved in the output directory.")