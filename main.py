import cv2
from videoProcessing import DashCamPreprocessor


video_file = "/Users/lvllvl/Dev/dev_SLAM/SLAM/data/train.mp4"
cap = cv2.VideoCapture( video_file )

frameNum = 90
while True:
    success, frame = cap.read()
    if not success:
        break
    frameNum += 1

    print(frameNum)
    # Process the frame: GrayScale
    frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

    # Process the frame: Blur, apply blur to reduce noise
    blurred_video = cv2.medianBlur( frame, 5 )
    binary_video = cv2.threshold( blurred_video, 127, 255, cv2.THRESH_BINARY )[1]
    
    # cv2.imshow( str(frameNum), frame)
    cv2.imshow( str(frameNum),blurred_video )
    # cv2.imshow( str(frameNum), binary_video[1] )
    cv2.waitKey(250)
    if frameNum == 100:
        break


# # threshold the video to make it black and white
# binary_video = cv2.threshold( blurred_video, 127, 255, cv2.THRESH_BINARY )[1]

# # Return the binary image
# return binary_video
# return src
# create a new instance of the DashCamPreprocessor class



#### uncomment this 
# preprocessor = DashCamPreprocessor( "/Users/lvllvl/Dev/dev_SLAM/SLAM/data/train.mp4" )

# # Preprocess the video footage
# binary_video = preprocessor.preprocess()

# # Display the binary image 
# cv2.imshow( "Binary Video", binary_video ) 
# cv2.waitKey(0)