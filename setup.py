import os
import shutil

def clone_repo(repo_url, dest_path):
    if os.path.exists(dest_path):
        print(f"Directory {dest_path} already exists. Removing and re-cloning.")
        shutil.rmtree(dest_path)
    os.system(f"git clone {repo_url} {dest_path}")

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move_files(src, dst):
    if not os.path.exists(src):
        print(f"Source directory {src} does not exist.")
        return
    for filename in os.listdir(src):
        src_file = os.path.join(src, filename)
        dst_file = os.path.join(dst, filename)
        try:
            shutil.move(src_file, dst_file)
        except Exception as e:
            print(f"Error moving {src_file} to {dst_file}: {e}")

def move_selected_files(src, dst, prefix=None, suffix=None):
    """
    Move files from src to dst based on prefix and/or suffix criteria.

    :param src: Source directory
    :param dst: Destination directory
    :param prefix: Prefix to filter files (optional)
    :param suffix: Suffix to filter files (optional)
    """
    if not os.path.exists(src):
        print(f"Source directory {src} does not exist.")
        return
    for filename in os.listdir(src):
        if prefix and not filename.startswith(prefix):
            continue
        if suffix and not filename.endswith(suffix):
            continue
        src_file = os.path.join(src, filename)
        dst_file = os.path.join(dst, filename)
        try:
            shutil.move(src_file, dst_file)
        except Exception as e:
            print(f"Error moving {src_file} to {dst_file}: {e}")



# Update filenames
def add_suffix_to_mask_filenames(image_dir, mask_dir, suffix):
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)

    for image_file in image_files:
        base, extension = os.path.splitext(image_file)
        expected_mask_file = f"{base}{suffix}{extension}"
        actual_mask_file = f"{base}{extension}"

        if actual_mask_file in mask_files:
            os.rename(os.path.join(mask_dir, actual_mask_file), os.path.join(mask_dir, expected_mask_file))
            print(f"Renamed {actual_mask_file} to {expected_mask_file}")


if __name__ == '__main__':
    # Clone repos
    clone_repo('https://github.com/lvllvl/SLAM.git', 'SLAM')
    clone_repo('https://github.com/commaai/comma10k.git', 'comma10k')
    
    # Create directories for training and validation
    create_directory('dataset_root/train/images')
    create_directory('dataset_root/train/masks')
    create_directory('dataset_root/val/images')
    create_directory('dataset_root/val/masks')

    # Move images and masks to training folder
    move_files('comma10k/imgs', 'dataset_root/train/images')
    move_files('comma10k/masks', 'dataset_root/train/masks')

    # Move images and masks to validation folder
    move_files('comma10k/imgs2', 'dataset_root/val/images')
    move_files('comma10k/masks2', 'dataset_root/val/masks')

    # Move SLAM *.py files to the dataset_root folder
    move_selected_files( 'SLAM/', 'dataset_root/', prefix=None, suffix='.py' )

    # List directories to verify
    print(os.listdir('.'))
    print(os.listdir('dataset_root'))

    # Rename mask files 
    add_suffix_to_mask_filenames('/content/dataset_root/train/images', '/content/dataset_root/train/masks', '_mask')