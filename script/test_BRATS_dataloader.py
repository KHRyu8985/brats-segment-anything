from src.data.BratsDataset import get_train_val_datasets
from monai.data import DataLoader
train_set, val_set = get_train_val_datasets(["data/BRATS_PED", "data/BRATS_GLI", "data/BRATS_MET", "data/BRATS_MEN"])

 # Create DataLoaders
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

print(f"Total training samples: {len(train_set)}")
print(f"Total validation samples: {len(val_set)}")

for i, data in enumerate(train_loader):
    img, seg, bbox = data
    print(f"Image shape: {img.shape}, Segmentation shape: {seg.shape}")
    print(f"Bounding box: {bbox}")
    if i == 0:
        break