import cv2

# Create a VideoCapture object and read from input file
class DashCamPreprocessor:

    def __init__(self, video_file):
        self.video_file = video_file
        # self.cap = cv2.VideoCapture(video_file)

    def preprocess( self ):
        # apply grayscale to reduce the amount of data that needs to be processes
        grayscale_video = cv2.cvtColor(self.video_file, cv2.COLOR_BGR2GRAY)

        # apply blur to reduce noise
        blurred_video = cv2.medianBlur( grayscale_video, 5 )

        # threshold the video to make it black and white
        binary_video = cv2.threshold( blurred_video, 127, 255, cv2.THRESH_BINARY )[1]

        # Return the binary image
        return binary_video