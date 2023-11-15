import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np

def visualize_data(image_dir, mask_dir, num_samples=5):
    image_files = os.listdir(image_dir)
    random.shuffle(image_files)
    image_files = image_files[:num_samples]

    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file.replace('.jpg', '_mask.jpg'))  # adjust mask file extension if needed

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title('Mask')
        plt.axis('off')

        plt.show()


def compute_image_statistics(image_dir):
    image_files = os.listdir(image_dir)
    stats = {'mean': [], 'std': []}
    
    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        img = Image.open(img_path)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        
        stats['mean'].append(np.mean(img_array))
        stats['std'].append(np.std(img_array))

    overall_mean = np.mean(stats['mean'])
    overall_std = np.mean(stats['std'])

    print(f"Overall Mean: {overall_mean}")
    print(f"Overall Std Dev: {overall_std}")


def analyze_label_distribution(mask_dir):
    mask_files = os.listdir(mask_dir)
    label_counts = {}

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)

        unique, counts = np.unique(mask_array, return_counts=True)
        for label, count in zip(unique, counts):
            label_counts[label] = label_counts.get(label, 0) + count

    total_pixels = sum(label_counts.values())
    for label, count in label_counts.items():
        print(f"Label {label}: {count / total_pixels * 100:.2f}%")




if __name__ == '__main__':
    visualize_data('/content/dataset_root/train/images', '/content/dataset_root/train/masks')
    compute_image_statistics('/content/dataset_root/train/images')
    analyze_label_distribution('/content/dataset_root/train/masks')