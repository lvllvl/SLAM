import torch
from torch import optim
from model import UNet
from dataloader import get_dataloaders

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
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loader, val_loader = get_dataloaders(
        train_dir='dataset_root/train/images',
        train_maskdir='dataset_root/train/masks',
        val_dir='dataset_root/val/images',
        val_maskdir='dataset_root/val/masks',
        batch_size=32
    )

    num_epochs = 25  # Number of epochs to train for

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate( model, val_loader, criterion, device )
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # add validation loop and early stopping logic

if __name__ == '__main__':
    main()
