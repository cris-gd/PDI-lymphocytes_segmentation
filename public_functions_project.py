import numpy as np
import cv2 as cv

# ---------------------------------------------------------
#  Public functions
# ---------------------------------------------------------

def Evaluate_Segmentation(labelmap_test, labelmap_gt):
    # Function to evaluate a labelmap against a GT reference
    # The number of levels of the input labelmap is arbitrary 
    # There is no correspondence between labels of both input labelmaps
    # Background is always represented by 0 value
    # It returns a final Dice coefficient

    dice_value_direct = labelmap_DC(labelmap_gt, labelmap_test)
    dice_value_inverse = labelmap_DC(labelmap_test, labelmap_gt)
    out_dice = (dice_value_direct+dice_value_inverse)/2
    # print('   Final Dice value: %f ' % out_dice)
    return out_dice

def Evaluate_Detection(n_CD8cells, labelmap_gt_CD8):
    # Evaluate the number of detected CD8 cells in the gt labelmap
    num_labels_gt_cd8, _ = cv.connectedComponents(labelmap_gt_CD8)
    num_labels_gt_cd8 = num_labels_gt_cd8 - 1 # We remove the background label
    return np.abs(n_CD8cells - num_labels_gt_cd8)

def most_frequent_element(array,binary_mask):
    # Mask the array using the binary mask
    masked_vector = array[binary_mask==1]
    # Find the most frequent element within the masked region
    values, counts = np.unique(masked_vector, return_counts=True)
    return values[np.argmax(counts)] 

def labelmap_DC(labelmap_1, labelmap_2):
    # One-direction Dice coefficient
    # Label 0 must always mean background 

    labels_1 = np.unique(labelmap_1)
    num_labels_1 = len(labels_1)
    labels_2 = np.unique(labelmap_2)
    num_labels_2 = len(labels_2)

    if num_labels_1==1 and labels_1[0]==0: # Just background in labelmpap_1
        if num_labels_2==1 and labels_2[0]==0: # Both labelmaps are empty
            out_dice = 1
        else: # labelmpap_1 is empty and labelmap_2 is not
            out_dice = 0
    else: # labelmap_1 is not empty, it has some object instances
        if num_labels_2==1 and labels_2[0]==0: # labelmap_2 is empty
            out_dice = 0
        else: # Both labelmaps are not empty
            out_dice_vector = np.zeros(num_labels_1)
            for i in range(0,num_labels_1):
                if labels_1[i]!=0:
                    mask_i = np.uint8(0*labelmap_1)
                    mask_i[labelmap_1==labels_1[i]] = 1
                    label_most_freq = most_frequent_element(labelmap_2,mask_i)
                    mask_2 = 0*labelmap_2
                    mask_2[labelmap_2==label_most_freq] = 1

                    # f, axarr = plt.subplots(2,2)
                    # axarr[0,0].imshow(labelmap_1)
                    # axarr[0,1].imshow(mask_i)
                    # axarr[1,0].imshow(labelmap_2)
                    # axarr[1,1].imshow(mask_2)
                    # plt.show()

                    if label_most_freq==0:
                        out_dice_vector[i] = 0
                    else:
                        out_dice_vector[i] = binary_DC(mask_i, mask_2)
            out_dice = out_dice_vector[1:].mean()
    return out_dice

def binary_DC(binary_img_1, binary_img_2):
    # Auxiliary function to calculate Dice coefficient from two binary images
    # The values of these MxN input images has to be 0 and 1, represented as uint8 type

    TP = np.count_nonzero(np.bitwise_and(binary_img_1, binary_img_2))
    TN = np.count_nonzero(np.bitwise_and(1-binary_img_1, 1-binary_img_2)) # Not used in this function. Included for completeness
    FP = np.count_nonzero(np.bitwise_and(binary_img_1, 1-binary_img_2))
    FN = np.count_nonzero(np.bitwise_and(1-binary_img_1, binary_img_2))
    dice_value = 2*TP/(2*TP+FP+FN)
    return dice_value

def create_random_LUT(num_channels=3):
    # Function to create a random false color list, used to represent a labelmap in a more appealing form
    # Please, feel free to change this code
    # The output list will be used as input parameter in fusion_labelmap() function

    my_random_lut = np.zeros([256,1,num_channels],dtype=np.uint8)
    for i in range(num_channels):
        one_channel_lut = np.uint8(np.arange(256))
        np.random.shuffle(one_channel_lut[1:])
        my_random_lut[:,0,i] = one_channel_lut
    return my_random_lut

def Fuse_Labelmap(base_img, labelmap, false_color_list, alpha, showQ, window_title):
    # Function to fusion an input gray-level image, named as base image, 
    # with its corresponding segmentation labelmap, shown over this base image as a semitransparent color layer
    # Its input parameter are:
        # base_img: : MxN gray level input image that serves as background of the final fused result
        # labelmap: MxN image input labelmap to be overlaid with previous image
        # false_color_list: list of false colors used for coloring the input labelmap
        # alpha: value of alpha parameter controlling the transparency level of the input labelmap
        # showQ: boolean variable to control if a window is opened, showing the fused image, 
               # with a slider control to modify the transparency coefficient
        # window_title: the title for the opened window, when applicable
    # It returns the fusioned image and the code of the key pressed to close the window, 
    # if the input variable showQ was True

    base_img_norm_8U = cv.normalize(base_img,None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)
    base_img_rgb = base_img_norm_8U

    # Apply false colors to the given labels
    list_labels = np.unique(labelmap)
    [rows, cols] = labelmap.shape
    labelmap_false_color = np.zeros((rows,cols,3),dtype=np.uint8)
    for label_i in list_labels:
        labelmap_false_color[labelmap==label_i,:] = false_color_list[label_i]

    fus_img = cv.addWeighted(base_img_rgb,1-alpha,labelmap_false_color,alpha,0.0)
    if showQ:
        [rows, cols] = labelmap.shape
        width_window = 800
        cv.namedWindow(window_title, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_title,width_window,round(width_window*rows/cols))
        alpha_slider_max = 100

        def onchange_any(val):
            val_alpha = cv.getTrackbarPos('Alpha blending coef', window_title)
            val_alpha = val_alpha/alpha_slider_max
            fus_img_aux = cv.addWeighted(base_img_rgb,1-val_alpha,labelmap_false_color,val_alpha,0.0)
            cv.imshow(window_title,fus_img_aux)

        cv.createTrackbar('Alpha blending coef',window_title,int(alpha*alpha_slider_max),
            alpha_slider_max,onchange_any)
        onchange_any(0)
        key_pressed = cv.waitKey()
        cv.destroyAllWindows()
    else:
        key_pressed = 0
    return fus_img, key_pressed

# ---------------------------------------------------------
#  End of public functions
# ---------------------------------------------------------
