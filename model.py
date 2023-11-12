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
import torch
import torch.nn as nn
import torch.nn.functional as F 

class UNet( nn.Module ):

    def up_conv( self, in_channels, out_channels ):
        return nn.ConvTranspose2d( in_channels, out_channels, kernel_size=2, stride=2 )
    
    def __init__( self ):
        super( UNet, self ).__init__()

        # Contracting Path ( Encoder )
        self.enc_conv0 = self.conv_block( 3, 64 )
        self.enc_conv1 = self.conv_block( 64, 128 )
        self.enc_conv2 = self.conv_block( 128, 256 )
        self.enc_conv3 = self.conv_block( 256, 512 )
        self.bottleneck = self.conv_block( 512, 1024 )

        # Expansive Path ( Decoder )
        self.up_conv3 = self.up_conv( 1024, 512 )
        self.dec_conv3 = self.conv_block( 512 + 512, 512 )
        self.up_conv2 = self.up_conv( 512, 256 )
        self.dec_conv2 = self.conv_block( 512, 256 )
        self.up_conv1 = self.up_conv( 256, 128 )
        self.dec_conv1 = self.conv_block( 256, 128 )
        self.up_conv0 = self.up_conv( 128, 64 )
        self.dec_conv0 = self.conv_block( 128, 64 )

        # Final output layer
        self.final_conv = nn.Conv2d( 64, NUM_CLASSES, kernel_size=1 )

    def conv_block( self, in_channels, out_channels ):
        return nn.Sequential( 
            nn.Conv2d( in_channels, out_channels, kernel_size=3, padding=1 ),
            nn.ReLU( inplace=True ),
            nn.Conv2d( out_channels, out_channels, kernel_size=3, padding=1 ),
            nn.ReLU( inplace=True )
        )
    
    def forward( self, x ):

        # Contracting Path ( Encoder )
        enc0 = self.enc_conv0( x )
        x = F.max_pool2d( enc0, 2 )
        enc1 = self.enc_conv1( x )
        x = F.max_pool2d( enc1, 2 )
        enc2 = self.enc_conv2( x )
        x = F.max_pool2d( enc2, 2 )
        enc3 = self.enc_conv3( x ) 
        x = F.max_pool2d( enc3, 2 )

        x = self.bottleneck( x )

        # Expansive Path ( Decoder )
        x = self.up_conv3( x )
        x = torch.cat( (x, enc3), dim=1 )
        x = self.dec_conv3( x )

        x = self.up_conv2( x ) 
        x = torch.cat( (x, enc2 ), dim=1 )
        x = self.dec_conv2( x )

        x = self.up_conv1( x )
        x = torch.cat( (x, enc1), dim=1 )
        x = self.dec_conv1( x )

        x = self.up_conv0( x )
        x = torch.cat( (x, enc0 ), dim=1 )
        x = self.dec_conv0( x )

        x = self.final_conv( x )
        return x

unet_model = UNet()




