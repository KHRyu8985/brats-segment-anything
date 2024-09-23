import autorootcwd
import cc3d
import nibabel as nib
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from intensity_normalization.normalize import nyul
import click

#THRESHOLD 
THRESHOLD = 1000

def process_label(input_path):
    # Load the NIfTI file
    nii_img = nib.load(input_path)
    seg = nii_img.get_fdata()

    # Create an empty array for the instance segmentation map
    instance_seg = np.zeros_like(seg, dtype=np.uint16)

    # Process each label
    for label in [1, 2, 3]:
        # Extract the binary mask for the current label
        mask = (seg == label)
        
        # Perform connected component analysis
        #labeled = cc3d.connected_components(mask, connectivity=26)
        labeled = cc3d.dust(mask, threshold=THRESHOLD, connectivity=18)
        labeled, N = cc3d.largest_k(labeled, k=3, return_N=True)
        
        print(f"File: {input_path}, Label: {label}, N: {N}")
        
        # Add the processed label to the instance segmentation map
        instance_seg[labeled > 0] = labeled[labeled > 0] + instance_seg.max()

    instance_seg, N = cc3d.largest_k(instance_seg, k=5, return_N=True)

    # Create a new NIfTI image with the instance segmentation map
    instance_nii = nib.Nifti1Image(instance_seg, nii_img.affine, nii_img.header)

    # Generate output path
    output_path = input_path.replace('-seg.nii.gz', '-instance_seg.nii.gz')

    # Save the instance segmentation map as a NIfTI file
    nib.save(instance_nii, output_path)

    print(f"Instance segmentation completed and saved as '{output_path}'")
    stats = cc3d.statistics(instance_seg)
    print('Number of voxels:')
    print(stats['voxel_counts'])


def normalize_modality(modality, files):
    # Use the first 15 images as templates
    template_files = files[:15]
    template_imgs = []
    template_masks = []

    print(f"Using the first 15 subjects as templates for {modality} normalization")

    for template_file in template_files:
        template_img = nib.load(template_file)
        template_data = template_img.get_fdata()
        template_mask = template_data > 0
        template_imgs.append(template_img)
        template_masks.append(template_mask)
        print(f"Added {template_file} as a template")

    # Initialize and train the Nyul normalizer
    normalizer = nyul.NyulNormalize(output_min_value=0, output_max_value=1)
    print(f"Training Nyul normalizer for {modality}...")
    normalizer.fit(template_imgs, masks=template_masks)

    # Process each file
    for input_file in tqdm(files, desc=f"Normalizing {modality} files"):
        target_nii = nib.load(input_file)
        target_data = target_nii.get_fdata()
        brain_mask = target_data > 0
        
        normalized_data = normalizer(target_data, mask=brain_mask)
        normalized_data = normalized_data * brain_mask
        
        output_path = input_file.replace(f'-{modality}.nii.gz', f'-{modality}_normalized.nii.gz')
        normalized_nii = nib.Nifti1Image(normalized_data, target_nii.affine, target_nii.header)
        nib.save(normalized_nii, output_path)
        
        print(f"Normalized {modality} image saved as '{output_path}'")

    print(f"All {modality} files normalized successfully.")

@click.command()
@click.option('--data-folder', default='data/BRATS_PED', type=click.Path(exists=True), help='Path to the dataset folder.')
def main(data_folder):
    """
    Process and normalize dataset images.
    """
    # Process label files (clean segmentation and transform to instance segmentation task)
    label_files = glob(os.path.join(data_folder, '**', '*-seg.nii.gz'), recursive=True)
    for input_file in tqdm(label_files, desc="Processing label files"):
        process_label(input_file)
    print("All label files processed successfully.")

    # Normalize modalities
    modalities = ['t1c', 't1n', 't2f', 't2w']
    for modality in modalities:
        files = glob(os.path.join(data_folder, '**', f'*-{modality}.nii.gz'), recursive=True)
        if files:
            normalize_modality(modality, files)
        else:
            print(f"No files found for modality: {modality}")

    print("All modalities normalized successfully.")

if __name__ == '__main__':
    main()
