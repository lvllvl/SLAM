from config import (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    NUM_CLASSES,
    MODEL_SAVE_NAME,
    SAVE_FREQUENCY,
)
import torch.nn as nn

class UNet( nn.Module ):

    def __init__( self ):
        super( UNet, self ).__init__()
        # ex of 1st conv layer (Unet)
        self.conv1 = nn.Conv2d( 3, 64, kernel_size=3, stride=1, padding=1 )
        self.relu1 = nn.ReLU( inplace=True )
        self.conv2 = nn.Conv2d( 64, 64, kernel_size=3, stride=1, padding=1 )
        self.relu2 = nn.ReLU( inplace=True )
        self.maxpool1 = nn.MaxPool2d( kernel_size=2, stride=2 )

    def forward( self, x ):
        # Layers to input
        x = self.conv1( x )
        x = self.relu1( x )
        x = self.conv2( x )
        x = self.relu2( x )
        x = self.maxpool1( x )
        #  more operations 
        
        return x 

