import cv2 
import glob
import random 

def load_crater_data_wrapper(method='scaled'):
    
    # set origin path for the images
    src = 'crater_data/images/normalized_images_' + method
    
    # this dict helps to create binary labels for the pictures
    labels_dict = {'crater': 1, 'non-crater': 0}
    
    samples = []
    # get all images file paths
    for src_filename in glob.glob(src + '/*/*.jpg'):
        # extract info from file path
        pathinfo = src_filename.split('/')
        img_type = pathinfo[-2] # crater or non-crater
        filename = pathinfo[-1] # the actual name of the jpg
        
        # read the grayscale version of the image, 
        # and normalize its values to be between 0 and 1
        img = cv2.imread(src_filename, cv2.IMREAD_GRAYSCALE) / 255.0
        
        # reshape the data structure to be a 1-D column vector
        img = img.flatten().reshape((len(img)**2, 1))
        
        # include the image data and its label into the sample list
        samples.append((img, labels_dict[img_type]))
    
    # We have to shuffle the order before splitting between training data
    # and test data
    random.shuffle(samples)
    
    # determine slices for training and test data
    splitpos = int(len(samples) * 0.7)
    return samples[:splitpos], samples[splitpos:]
    