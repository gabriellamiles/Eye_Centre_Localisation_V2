import cv2

def review_data(img_filepath, eye_coords, eye_preds):

    eye_coords = (int(eye_coords[0]), int(eye_coords[1]))
    eye_preds = (int(eye_preds[0]), int(eye_preds[1]))

    # load image
    print(img_filepath)
    im = cv2.imread(img_filepath)

    # cv2.namedWindow(img_filepath)
    
    # draw predictions on image
    cv2.circle(im, eye_coords, 2, (0,255,0), -1)

    cv2.circle(im, eye_preds, 2, (255,0,0), -1)

    # display image
    cv2.imshow('image', im)

    retrain_image = 0
    print("Ground truth (green): " + str(eye_coords))
    print("Prediction: (blue)" + str(eye_preds))

    while(1):

        cv2.imshow('image', im)

        k = cv2.waitKey(20) & 0xFF

        if k == ord('n'): # press n
            # skip/go to next image
            break

        elif k == ord('y'): # press y to add point 
            retrain_image = 1
            break

        elif k == ord('r'): # press r to denote relabelling necessary
            retrain_image = 2
            break

    cv2.destroyAllWindows()
    
    return retrain_image