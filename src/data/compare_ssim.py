import os
import cv2
# from skimage import io
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# initialise key filepaths
img_folders = os.path.join(os.getcwd(), "data", "processed", "mnt", "eme2_square_imgs")
# participant_folders = [os.path.join(img_folders, i) for i in os.listdir(img_folders)]
participant_folders = [os.path.join(img_folders, i) for i in ["053"]]#, "007", "010", "056", "064", "075", "201"]]
# print(participant_folders)

for participant in participant_folders:
    for trial in ["0"]:# "1"]:

        # if os.path.exists(os.path.basename(participant) + "_" + trial + '.csv'):
        #     continue

        # Set the directory containing the images
        trial_folder = os.path.join(participant, trial)
        print(trial_folder)

        if not os.path.exists(trial_folder):
            print(trial_folder + " does not exist at this mount point.")
            continue

        #  List all image files in the directory
        image_files = sorted([f for f in os.listdir(trial_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
        image_files = image_files[:30000]
        image_files = image_files[::10]
        # print(image_files)

        # Set the reference image filename
        reference_image_filename = image_files[2000]

        # Load the reference image
        reference_image = cv2.imread(os.path.join(trial_folder, reference_image_filename))

        # Initialize a dictionary to store SSIM scores
        ssim_scores = {}
        count = 0

        # Compare the SSIM of the reference image to each image in the directory
        for image_file in image_files:
            
            if image_file != reference_image_filename:
                # Load the current image
                current_image = cv2.imread(os.path.join(trial_folder, image_file))

                # Convert images to grayscale for SSIM comparison (assuming they have the same size)
                reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
                current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

                # Calculate SSIM
                try:
                    score = ssim(reference_image_gray, current_image_gray)

                    # Store the SSIM score in the dictionary
                    ssim_scores[image_file] = score
                except:
                    pass
                
                print(str(count) + "/" + str(len(image_files)) + "   (" + str(score) + ")")

            count+=1

        # Sort the images by SSIM score in descending order
        sorted_images = sorted(ssim_scores.items(), key=lambda x: x[1], reverse=True)

        # Keep the top 50 most dissimilar images
        top_50_dissimilar_images = [image[0] for image in sorted_images][-80:]

        # Create a DataFrame to store the results
        df = pd.DataFrame({'Image File': top_50_dissimilar_images, 'SSIM Score': [ssim_scores[image] for image in top_50_dissimilar_images]})

        # Save the DataFrame to a CSV file
        
        df.to_csv(os.path.basename(participant) + "_" + trial + '.csv', index=False)

        print("Top 50 most dissimilar images saved to: " + os.path.basename(participant) + "_" + trial + '.csv')