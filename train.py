from dataloader import get_dataloaders

def main():
    train_loader, val_loader = get_dataloaders(
        train_dir='dataset_root/train/images',
        train_maskdir='dataset_root/train/masks',
        val_dir='dataset_root/val/images',
        val_maskdir='dataset_root/val/masks',
        batch_size=32
    )

    # ... training loop ...

if __name__ == '__main__':
    main()
