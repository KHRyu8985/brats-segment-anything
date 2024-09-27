import autorootcwd
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai import transforms
from tqdm import tqdm
from src.data.BratsDataset import post_process
from src.segment_anything.modeling.image_encoder import ImageEncoderViT
from src.segment_anything.modeling.prompt_encoder import PromptEncoder
from src.segment_anything.modeling.mask_decoder import MaskDecoder
from src.segment_anything.modeling import TwoWayTransformer

import torch.nn as nn
import torch
import torch.nn.functional as F

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

image_encoder = ImageEncoderViT(img_size=64, patch_size=4, in_chans=1)
prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(16,16), 
                               input_image_size=(64,64), mask_in_chans=1)
mask_decoder = MaskDecoder(transformer_dim=256, transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),)


batch_size = 2
channel = 0

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

train_ds = DecathlonDataset(
    root_dir='data/Task01_BrainTumour',
    task="Task01_BrainTumour",
    section="training",  # validation
    cache_rate=0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=4,
    download=False,  # Set download to True if the dataset hasnt been downloaded yet
    seed=0,
    transform=train_transforms,
)

print(f"Length of training data: {len(train_ds)}")  # this gives the number of patients in the training set


train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True
)

for epoch in range(100):
    epoch_loss = 0.0
    for step, (image, gt2D, boxes) in enumerate(tqdm(train_loader)):

data = next(iter(train_loader))
img, seg, bbox = data[0], data[1], data[2]

print(bbox)

image_embedding = image_encoder(img)
print('Image Embedding: ', image_embedding.shape)

sparse_embeddings, dense_embeddings = prompt_encoder(points=None, boxes=bbox, masks=None)
print('Sparse embedding:', sparse_embeddings.shape, 'Dense:', dense_embeddings.shape)

# Decode the mask using the mask decoder
low_res_logits, _ = mask_decoder(image_embedding, 
                                prompt_encoder.get_dense_pe(), 
                                sparse_embeddings, 
                                dense_embeddings,
                                multimask_output=False)

print('Final output:', low_res_logits.shape)
