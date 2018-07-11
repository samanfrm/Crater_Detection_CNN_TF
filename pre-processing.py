import cv2
import glob
import os

src = 'crater_data/images/tile3_24'
dsta = 'crater_data/images/normalized_images_padded'
dstb = 'crater_data/images/normalized_images_scaled'

# create new directories if necessary
for dst in [dsta, dstb]:
    for imgtype in ['crater', 'non-crater']:
        tgdir = os.path.join(dst, imgtype)
        if not os.path.isdir(tgdir):
            os.makedirs(tgdir)

BGCOLOR = [0, 0, 0]
for src_filename in glob.glob(src + '/*/*.jpg'):
    pathinfo = src_filename.split('/')
    img_type = pathinfo[-2] # crater or non-crater
    filename = pathinfo[-1] # the actual name of the jpg
    
    dsta_filename = os.path.join(dsta, img_type, filename)
    dstb_filename = os.path.join(dstb, img_type, filename)

    # read the original image and get size info
    src_img = cv2.imread(src_filename)
    height, width = src_img.shape[:2]
    
    # get size of the padding, apply padding and write to disk
    brd = int((200 - height) / 2)
    padded_img = cv2.copyMakeBorder(src_img, brd, brd, brd, brd,
                                    cv2.BORDER_CONSTANT, value=BGCOLOR)
    cv2.imwrite(dsta_filename, padded_img)
    
    # resize image, normalize and write to disk
    scaled_img = cv2.resize(src_img, (200, 200))
    cv2.normalize(scaled_img, scaled_img, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(dstb_filename, scaled_img)