import autorootcwd
from monai.apps import DecathlonDataset
from monai import transforms
from monai.data import DataLoader
import numpy as np
import cc3d
import matplotlib.pyplot as plt
import os
import torch
import random

channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"

def post_process(data):
    image, label = data['image'], data['label']
    # make label binary (whole tumour)
    label = (label > 0).float()
    label_instance = create_instance_segmentation(label) # semantic to instance
    # Process instance segmentation
    N = label_instance.max() # number of instances
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
    
    labeled = cc3d.dust(mask, threshold=20, connectivity=8)  # delete small disconnected components
    labeled, _ = cc3d.largest_k(labeled, k=10, return_N=True)  # keep only the largest 3 components
    
    # Update label_instance using tensor operations
    label_instance += torch.from_numpy(labeled).to(label_instance.device) * (labeled > 0)
    
    return label_instance

def generate_bounding_box(seg, bbox_shift=5):
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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(3.0, 3.0, 2.0), mode=("bilinear", "nearest")),
        transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 64)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 1), random_size=False),
        transforms.Lambdad(keys=["image", "label"], func=lambda x: x.squeeze(-1)),
        transforms.Rotate90d(keys=["image", "label"], k=1),  # Rotate 90 degrees
        transforms.Lambda(post_process),
    ]
)

if __name__ == '__main__':

    batch_size = 2

    train_ds = DecathlonDataset(
        root_dir='data/Task01_BrainTumour',
        task="Task01_BrainTumour",
        section="training",  # validation
        cache_rate=0,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=4,
        download=True,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=train_transforms,
    )

    print(f"Length of training data: {len(train_ds)}")  # this gives the number of patients in the training set


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True
    )

    val_ds = DecathlonDataset(
        root_dir='data/Task01_BrainTumour',
        task="Task01_BrainTumour",
        section="validation",
        cache_rate=0,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=4,
        download=True,  # Set download to True if the dataset hasnt been downloaded yet
        seed=0,
        transform=train_transforms,
    )
    print(f"Length of training data: {len(val_ds)}")

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True
    )

    # Visualize a few samples
    visualization_folder = "visualization"
    if not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)

    for i in range(100):  # Visualize 5 random samples
        img, seg, bbox = train_ds[np.random.randint(len(train_ds))]
        if sum(bbox) == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(25, 15))
        axes[0].imshow(img[0], cmap="gray")
        axes[0].axis('off')  # Turn off axis for the first plot
        
        axes[1].imshow(img[0], cmap="gray")
        axes[1].axis('off')  # Turn off axis for the second plot
        show_mask(seg[0].cpu().numpy(), axes[1])
        show_box(bbox, axes[1])
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(os.path.join(visualization_folder, f"sample_{i}.png"))
        plt.close(fig)  # Close the figure to free up memory

