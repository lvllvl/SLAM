import unittest
from videoProcessing import DashCamPreProcessor

class TestDashCamPreProcessor(unittest.TestCase):

    def test_openVideo( self ):

        unittest.assertTrue( DashCamPreProcessor.self.video_file == "/Users/lvllvl/Dev/dev_SLAM/SLAM/data/train.mp4")