import cv2
import numpy as np

'''
Create a VideoCapture object and read from input file
Process the video footage 
Return the each frame processed and saved as a binary image
'''
class DashCamPreprocessor:

    def __init__(self, video_file):
        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)

    def preprocess( self ):
        '''
        Preprocess the video footage:
            open video file
            break video into frames
            convert each frame to grayscale
            convert each frame to blur
            convert each frame to binary 
        '''

        # TODO: #2 Figure out how to save frames  and for how long, e.g., discard after optical flow?
        frameNum = 1 
        while True:
            success, frame = self.cap.read()
            if not success:
                break # TODO: #1 create a way to log this error and in project in general
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