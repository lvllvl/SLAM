# Paths
TRAIN_IMAGE_DIR = '/content/dataset_root/train/images'
TRAIN_MASK_DIR = '/content/dataset_root/train/masks'
VAL_IMAGE_DIR = '/content/dataset_root/val/images'
VAL_MASK_DIR = '/content/dataset_root/val/masks'
CHECKPOINT_DIR = '/content/drive/MyDrive/Colab Notebooks/unet-checkpoints'
LOG_DIR = 'path/to/logs'

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 25
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES = 5  # Update based on the number of classes in your dataset

# Model settings
MODEL_NAME = 'UNet'
MODEL_SAVE_NAME = f'{MODEL_NAME}_best.pth'

# Data augmentation and preprocessing
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Training settings
SAVE_FREQUENCY = 5  # How often to save checkpoints
