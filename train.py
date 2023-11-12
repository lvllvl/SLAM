import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import UNet
from dataloader import get_dataloaders
import os
from config import (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    # NUM_CLASSES,
    MODEL_SAVE_NAME,
    SAVE_FREQUENCY,
)

# Now you can use these variables in your main script



def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate( model, dataloader, criterion, device ):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to( device ), masks.to( device )
            outputs = model( images )
            loss = criterion( outputs, masks )
            running_loss += loss.item()
    val_loss = running_loss / len( dataloader ) 
    return val_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = torch.nn.CrossEntropyLoss() # or any other appropriate loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Reduce LR on Plateau by 0.1 if val loss does not decrease for 2 epochs
    scheduler = ReduceLROnPlateau( optimizer, 'min', patience=2, factor=0.1, verbose=True )

    train_loader, val_loader = get_dataloaders(
        train_dir=TRAIN_IMAGE_DIR,
        train_maskdir=TRAIN_MASK_DIR,
        val_dir=VAL_IMAGE_DIR,
        val_maskdir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE
    )
    
    num_epochs = NUM_EPOCHS  # Number of epochs to train for
    best_val_loss = float('inf')
    checkpoint_dir = CHECKPOINT_DIR 
    save_frequency = SAVE_FREQUENCY  # Save every 5 epochs, checkpoint

    # Create checkpoint directory if directory DNE
    if not os.path.exists( checkpoint_dir ):
        os.makedirs( checkpoint_dir )

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate( model, val_loader, criterion, device )
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'model_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

        scheduler.step( val_loss )

        if (epoch + 1) % save_frequency == 0:
            periodic_checkpoint_path = os.path.join( checkpoint_dir, f'model_checkpoint_epoch_{epoch+1}.pth')
            # TODO: add the right path below
            torch.save( model.state_dict(), f'/content/drive/MyDrive/Colab Notebooks/unet-checkpoints/model_checkpoint_epoch_{epoch+1}.pth')
            print( f"Periodic checkpoint saved: Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

if __name__ == '__main__':
    main()
