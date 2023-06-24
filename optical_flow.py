# !/usr/bin/env python3

import cv2
import numpy as np

def lucas_kanade_method( video_path ):

    # Read the video file
    cap = cv2.VideoCapture( video_path )

    # Parameters for Lucas Kanade optical flow
    lk_params = dict( 
        winSize = (15,15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Create random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_frame = cv2.cvtColor( old_frame, cv2.COLOR_BGR2GRAY )
    p0 = cv2.goodFeaturesToTrack( 
        old_frame, 
        mask = None, 
        **dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 ) 
        )
    # Create a mask image for drawing purposes
    mask = np.zeros_like( old_frame )

    frame_num = 1
    # while frame_num < 50:
    while True:
        # read the freame 
        ret, frame = cap.read()
        if not ret:
            break

        print( "Frame number: ", frame_num)
    
        frame_gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

        #Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK( 
            old_frame, 
            frame_gray, 
            p0, 
            None, 
            **lk_params 
            )
        
        # print( "pi type: ", type(p1), "st type: ", type(st), "err type: ", type(err) )
        # break 

        

        # Select good points
        good_new = p1[ st == 1 ]
        good_old = p0[ st == 1 ]

        # draw the tracks
        for i, (new, old) in enumerate( zip( good_new, good_old )):

            a, b = new.ravel()
            c, d = old.ravel()

            a, b = int(a), int(b)
            c, d = int(c), int(d)
            
            mask = cv2.line( mask, (a,b), (c,d), color[i].tolist(), 2 )
            frame = cv2.circle( frame, (a,b), 5, color[i].tolist(), -1 )

        # Display the demo
        # print( "Frame size: ", frame[:, :, 0].shape )
        # print( "Mask size: ", mask.shape )
          
        img = cv2.add( frame[:,:,0], mask )
        cv2.imshow( "frame", img )
        k = cv2.waitKey( 25 ) & 0xFF

        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape( -1, 1, 2 )
        frame_num += 1

if __name__ == "__main__":
    lucas_kanade_method( "data/test.mp4" )