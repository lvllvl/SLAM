import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import UNet
from dataloader import get_dataloaders
import os
import time
from torch.profiler import profile, ProfilerActivity
from config import *

def validate( model, dataloader, criterion, device ):

    model.eval()
    running_loss = 0.0
    total_batches = len( dataloader )
    last_batch_idx = 0 # init the last batch index

    with torch.no_grad():
        for batch_idx, ( images, masks ) in enumerate( dataloader ):
            images, masks = images.to( device ), masks.to( device ).long()
            outputs = model( images )
            loss = criterion( outputs, masks )
            running_loss += loss.item()
            print( f'Validation - Batch { batch_idx + 1 } / { total_batches }, Current Loss: { loss.item():.4f}' ) 
            last_batch_idx = batch_idx # update the last batch index
    average_loss = running_loss / total_batches    
    return average_loss, last_batch_idx

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_batches = len( dataloader ) 
    last_batch_idx = 0 # init the last batch index

    for batch_idx, (images, masks) in enumerate(dataloader):
        last_batch_idx = batch_idx # update the last batch index
        images, masks = images.to(device), masks.to(device).long() # Convert masks to long to match model type

        # Forward pass
        outputs = model(images)

        # Squeeze the channel dimension if necessary
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print( f'Batch {batch_idx + 1 } / {total_batches }, Current Loss: {loss.item():.4f}' )

    return running_loss / total_batches 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    print(f'Model Summary:\n{model}')  # Model summary

    criterion = torch.nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)

    train_loader, val_loader = get_dataloaders(
        # DataLoader setup
        train_dir=TRAIN_IMAGE_DIR,
        train_maskdir=TRAIN_MASK_DIR,
        val_dir=VAL_IMAGE_DIR,
        val_maskdir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE
    )

    num_epochs = NUM_EPOCHS
    best_val_loss = float('inf')
    checkpoint_dir = CHECKPOINT_DIR
    save_frequency = SAVE_FREQUENCY

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_last_batch = None
        val_last_batch = None
        
        # Start profiler
        with profile( activities=[ ProfilerActivity.CPU, ProfilerActivity.CUDA ], record_shapes=True ) as prof:
            try:
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_last_batch = validate(model, val_loader, criterion, device)

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                print(f'Epoch {epoch+1}/{num_epochs}, Duration: {epoch_duration:.2f}s, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(checkpoint_dir, f'model_best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Best model saved at {checkpoint_path}")

                scheduler.step(val_loss)

                if (epoch + 1) % save_frequency == 0:
                    periodic_checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_epoch_{epoch+1}.pth')
                    torch.save(model.state_dict(), periodic_checkpoint_path)
                    print(f"Periodic checkpoint saved at {periodic_checkpoint_path}")

            except Exception as e:
                
                # decide which batch index to report based on where the error occured
                report_batch_idx = val_last_batch if val_last_batch is not None else train_last_batch
                error_message = f"Error during epoch {epoch+1}, Batch {report_batch_idx+1}"
                print( error_message )
        # Stop profiler
        print( prof.key_averages().table( sort_by="cuda_time_total", row_limit=10 ) )


if __name__ == '__main__':
    main()
