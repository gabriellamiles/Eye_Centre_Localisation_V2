import os
import cv2

if __name__ == '__main__':

    root_folder = os.getcwd()
    img_folder = os.path.join(root_folder, "data", "processed", "mnt1", "eme2_square_imgs", "011")

    # Create a directory to store the modified images
    output_directory = 'dataset_images'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define the number of pixels to cut from the top and bottom
    cut_pixels = 280

    # Define the scaling factor
    scaling_factor = 0.66

    # Loop through each subdirectory within the main directory
    for subdir in os.listdir(img_folder):
        subdir_path = os.path.join(img_folder, subdir)
        
        if os.path.isdir(subdir_path):
            # Create a subdirectory for modified images
            output_subdirectory = os.path.join(output_directory, os.path.basename(img_folder), subdir)
            if not os.path.exists(output_subdirectory):
                os.makedirs(output_subdirectory)
            
            # List all image files in the subdirectory
            image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            # Process each image in the subdirectory
            for image_file in image_files:
                image_path = os.path.join(subdir_path, image_file)
                
                # Read the image
                image = cv2.imread(image_path)
                
                # Get image dimensions
                height, width, _ = image.shape
                
                # Remove the top and bottom pixels
                modified_image = image[cut_pixels:height - cut_pixels, :]

                # Calculate the new dimensions after rescaling
                new_height = int(modified_image.shape[0] * scaling_factor)
                new_width = int(modified_image.shape[1] * scaling_factor)
                
                # Rescale the image
                rescaled_image = cv2.resize(modified_image, (new_width, new_height))
                
                # Save the modified and rescaled image to the output subdirectory
                output_path = os.path.join(output_subdirectory, image_file)
                cv2.imwrite(output_path, rescaled_image)

    print("Images with top and bottom {} pixels removed have been saved in the '{}' directory.".format(cut_pixels, output_directory))