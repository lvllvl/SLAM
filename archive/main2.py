import cv2
from videoProcessing import DashCamPreprocessor

preprocessor = DashCamPreprocessor( "/Users/lvllvl/Dev/dev_SLAM/SLAM/data/train.mp4" )

# Preprocess the video footage
binary_video = preprocessor.preprocess()

# # Display the binary image 
# cv2.imshow( "Binary Video", binary_video ) 
# cv2.waitKey(0)