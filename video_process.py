#!/usr/bin/env python3

import cv2
import numpy as np
import os

class DashCamPreprocessor:

    '''
    Create a VideoCapture object and read from input file
    Process the video footage 
    Return the each frame processed and saved as a binary image

    [] create a function to delete all the frames in the frames folder
    '''
    def __init__(self, video_file):
        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)

    def delete_frames(self) -> bool:
       file_frames = os.listdir("frames") 
       for f in file_frames:
           os.remove("frames/" + f)

       file_frames = os.listdir("frames")
       return len(file_frames) == 0

    def preprocess( self ) -> bool:
        '''
        Preprocess the video footage:
            [x] open video file
            [x] break video into frames
            [x] convert each frame to grayscale
            [x] convert each frame to blur
            [] convert each frame to binary 
            [] save each frame as a binary image, or just as an image

        '''

        frame_num = 1 
        # while success:
        while frame_num < 11:
            success, frame = self.cap.read()
            if not success:
                break

            print("Frame number: ", frame_num)

            # Process the frame: GrayScale
            frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

            # Process the frame: Blur, apply blur to reduce noise
            blurred_video = cv2.medianBlur( frame, 5 )
            binary_video = cv2.threshold( blurred_video, 127, 255, cv2.THRESH_BINARY )[1]
            
            cv2.imshow( str(frame_num),blurred_video )
            # cv2.imshow( str(frame_num), binary_video[1] )
            cv2.waitKey(250)
            if frame_num == 10:
                break
            

            # save image 
            cv2.imwrite( "frames/frame%d.jpg" % frame_num, frame)
            
            # When everything done, release the capture
            frame_num += 1
        return True


if __name__ == "__main__":
    dashcam = DashCamPreprocessor( "data/train.mp4" )
    dashcam.preprocess()

    print( dashcam.delete_frames() ) 
    print("Done")