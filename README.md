# Crater_Detection_CNN_TF


Impacts and cratering on solar system objects is a fundamental and continuing process. Recognizing and
distinguishing impact craters is a difficult task even for expert observers, because of their varied sizes
and vast presence.

The main objective of this project is to train a Convolution Neural Network using Tensor Flow for recognizing and distinguishing craters on Martian surface images

Datasets:
Our dataset contains the images and ground-truth (gt) folders. The images folder includes
two images from the surface of Mars planet in the pgm format (tile3_24.pgm,
tile3_25.pgm). The size of these images are 1,700 by 1,700 pixels. Also, there are two subfolders named tile3_24 and tile3_25 on this folder which contains extracted crater and noncrater regions from the .pgm files. The crater subfolder contains the positive examples
(craters) in the jpg format. The non-crater sub-folder represents non-crater regions
extracted from the .pgm file. The negative example (non-crater) extracted randomly using a
uniform distribution. The negative samples with low standard deviation (less than 5) and
with significant overlapped area with craters (more than 25%) selected for this project. The
number of negative examples used was approximately the double of the positive ones.
