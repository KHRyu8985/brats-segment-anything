import autorootcwd
import os
import numpy as np
import torch
import cc3d
import matplotlib.pyplot as plt
from glob import glob
from monai import transforms
from monai.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset

import monai
from src.utils.registry import DATASET_REGISTRY

from monai.transforms import (
    Compose,
    RandSpatialCropd,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    Lambdad,
    LoadImaged,
    Orientationd,
    Rotate90d,
    Resized,
    RandCropByPosNegLabeld,
    Lambda,
    ScaleIntensityd,
    SqueezeDimd,
    ScaleIntensityRangePercentilesd
)

modalities = ['t1c', 't1n', 't2f', 't2w']


def plot_image_and_label(image, label):
    """
    Plot image and label side by side in a subplot.

    Args:
    image (torch.Tensor): Image tensor of shape [C, H, W]
    label (torch.Tensor): Label tensor of shape [H, W]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot image
    ax1.imshow(image.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Image')
    ax1.axis('off')

    show_mask(label.cpu().numpy(), ax1)

    # Plot label
    ax2.imshow(label.cpu().numpy(), cmap='viridis')
    ax2.set_title('Label')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def get_subject_id(filename):
    return os.path.basename(filename).split('-')[0]


def post_process(data):
    image, label = data['image'], data['label']
    # make label binary (whole tumour)
    label = (label > 0).float()
    label_instance = create_instance_segmentation(
        label)  # semantic to instance
    # Process instance segmentation
    N = label_instance.max()  # number of instances
    if N > 0:
        # generate random click
        idx = np.random.randint(1, N + 1)
        generated_segmentation = (label_instance == idx)
        generated_bbox = generate_bounding_box(generated_segmentation)
        generated_bbox = torch.tensor(generated_bbox).float()
    else:
        generated_segmentation = torch.zeros_like(label_instance)
        generated_bbox = torch.tensor((0, 0, 0, 0)).float()

    return (image, generated_segmentation, generated_bbox)


def create_instance_segmentation(label):
    if isinstance(label, torch.Tensor):
        label_numpy = label.cpu().numpy().squeeze()
    else:
        label_numpy = label

    # Create a zeros tensor with the same properties as label
    label_instance = torch.zeros_like(label)

    mask = (label_numpy == 1)

    # delete small disconnected components
    labeled = cc3d.dust(mask, threshold=100, connectivity=4)
    # keep only the largest 3 components
    labeled, _ = cc3d.largest_k(labeled, k=10, return_N=True)

    # Update label_instance using tensor operations
    label_instance += torch.from_numpy(labeled).to(
        label_instance.device) * (labeled > 0)

    return label_instance


def generate_bounding_box(seg, bbox_shift=20):
    """
    Generate a bounding box for a given segmentation mask using PyTorch
    Input seg shape: [1, h, w]
    """
    # Assert seg is not empty and is not zero everywhere
    assert seg is not None, "Segmentation mask is empty"
    assert seg.max() > 0, "Segmentation mask is zero everywhere"

    # Ensure seg is a torch tensor
    if not isinstance(seg, torch.Tensor):
        seg = torch.from_numpy(seg)

    # Remove the channel dimension and any extra dimensions
    seg = seg.squeeze()

    # Find non-zero indices
    y, x = torch.where(seg > 0)

    x_min = x.min().item()
    x_max = x.max().item()
    y_min = y.min().item()
    y_max = y.max().item()

    # Add perturbation to bounding box coordinates
    H, W = seg.shape
    x_min = max(0, x_min - torch.randint(0, bbox_shift + 1, (1,)).item())
    x_max = min(W, x_max + torch.randint(0, bbox_shift + 1, (1,)).item())
    y_min = max(0, y_min - torch.randint(0, bbox_shift + 1, (1,)).item())
    y_max = min(H, y_max + torch.randint(0, bbox_shift + 1, (1,)).item())

    return x_min, y_min, x_max, y_max


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="blue",
                 facecolor=(0, 0, 0, 0), lw=2))


def get_dataset(root_dir):
    seg = sorted(glob(os.path.join(root_dir, "**/*-seg.nii.gz"),
                 recursive=True), key=get_subject_id)
    t1c = sorted(glob(os.path.join(root_dir, "**/*-t1c.nii.gz"),
                 recursive=True), key=get_subject_id)
    t1n = sorted(glob(os.path.join(root_dir, "**/*-t1n.nii.gz"),
                 recursive=True), key=get_subject_id)
    t2f = sorted(glob(os.path.join(root_dir, "**/*-t2f.nii.gz"),
                 recursive=True), key=get_subject_id)
    t2w = sorted(glob(os.path.join(root_dir, "**/*-t2w.nii.gz"),
                 recursive=True), key=get_subject_id)

    # Verify that all lists have the same length and order
    assert len(seg) == len(t1c) == len(t1n) == len(t2f) == len(
        t2w), "Mismatch in number of files across modalities"

    for s, c, n, f, w in zip(seg, t1c, t1n, t2f, t2w):
        assert get_subject_id(s) == get_subject_id(c) == get_subject_id(n) == get_subject_id(f) == get_subject_id(w), \
            f"Mismatch in subject order: {s}, {c}, {n}, {f}, {w}"

    print(f"Found {len(seg)} subjects with all modalities in {root_dir}.")

    train_files = [{"image": [c, n, f, w], "label": _seg}
                   for c, n, f, w, _seg in zip(t1c, t1n, t2f, t2w, seg)]

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=0, upper=99.5, b_min=0, b_max=1, channel_wise=True),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=[
                               240, 240, 1], pos=1, neg=1, num_samples=4),
        Lambdad(keys=["image", "label"], func=lambda x: x.squeeze(-1)),
        Resized(keys=["image"], spatial_size=(1024, 1024),
                anti_aliasing=True, mode="nearest"),
        Resized(keys=["label"], spatial_size=(1024, 1024),
                anti_aliasing=False, mode="area"),
        Rotate90d(keys=["image", "label"], k=1),  # Rotate 90 degrees
        Lambda(post_process)
    ])

    volume_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=0.01)
    return volume_ds


def save_plot(image, label, bbox, index, folder):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot image
    ax1.imshow(image.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Image')
    ax1.axis('off')

    show_mask(label.cpu().numpy(), ax1)
    show_box(bbox, ax1)

    # Plot label
    ax2.imshow(label.cpu().numpy(), cmap='viridis')
    ax2.set_title('Label')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"plot_{index:03d}.png"))
    plt.close(fig)


def get_train_val_datasets(dataset_paths=[
        "data/BRATS_GLI",
        "data/BRATS_MET",
        "data/BRATS_MEN",
        "data/BRATS_PED"
    ]
):
    train_datasets = []
    val_datasets = []

    for dataset_path in dataset_paths:
        train_path = os.path.join(dataset_path, "Train")
        val_path = os.path.join(dataset_path, "Val")

        dataset_name = os.path.basename(dataset_path)
        print(f"Processing {dataset_name}...")

        train_ds = get_dataset(train_path)
        val_ds = get_dataset(val_path)

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    # Concatenate all train datasets
    combined_train_dataset = ConcatDataset(train_datasets)

    # Concatenate all validation datasets
    combined_val_dataset = ConcatDataset(val_datasets)

    return combined_train_dataset, combined_val_dataset



if __name__ == "__main__":
    datasets = [
        "data/BRATS_GLI/Train",
        "data/BRATS_MET/Train",
        "data/BRATS_MEN/Train",
        "data/BRATS_PED/Train"
    ]

    for dataset_path in datasets:
        dataset_name = os.path.basename(os.path.dirname(dataset_path))
        print(f"Processing {dataset_name}...")

        volume_ds = get_dataset(dataset_path)
        check_loader = DataLoader(volume_ds, batch_size=1)

        # Create visualization folder if it doesn't exist
        visualization_folder = f"script/visualization_{dataset_name}"
        os.makedirs(visualization_folder, exist_ok=True)

        # Save 50 plots for each dataset
        for i, data in enumerate(check_loader):
            if i >= 5:
                break
            img, seg, bbox = data
            print(f"Processing sample {i + 1}...")
            print(f"Image shape: {img.shape}, Segmentation shape: {seg.shape}")
            print(f"Bounding box: {bbox}")
            save_plot(img[0, 2], seg[0, 0], bbox[0], i, visualization_folder)

        print(
            f"Saved 50 plots for {dataset_name} in the '{visualization_folder}' directory.")

    print("All datasets processed.")


    # Usage
    combined_train_dataset, combined_val_dataset = get_train_val_datasets()

    # Create DataLoaders
    train_loader = DataLoader(combined_train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(combined_val_dataset, batch_size=1, shuffle=False)

    print(f"Total training samples: {len(combined_train_dataset)}")
    print(f"Total validation samples: {len(combined_val_dataset)}")

    # Create visualization folder if it doesn't exist
    visualization_folder = f"script/visualization_concat"
    os.makedirs(visualization_folder, exist_ok=True)

    # Save 50 plots for each dataset
    for i, data in enumerate(train_loader):
        if i >= 20:
            break
        img, seg, bbox = data
        print(f"Processing sample {i + 1}...")
        print(f"Image shape: {img.shape}, Segmentation shape: {seg.shape}")
        print(f"Bounding box: {bbox}")
        save_plot(img[0, 2], seg[0, 0], bbox[0], i, visualization_folder)

    print(
        f"Saved 50 plots for {dataset_name} in the '{visualization_folder}' directory.")

print("All datasets processed.")