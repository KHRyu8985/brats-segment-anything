import autorootcwd
import cc3d
import nibabel as nib
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from intensity_normalization.normalize import nyul
import click
import fnmatch  # Import fnmatch for pattern matching

# THRESHOLD 
THRESHOLD = 1000

def delete_dot_underscore_files(valdo_folder, dry_run=False):
    """
    Deletes files in the VALDO folder that start with '._sub'.

    Parameters:
    - valdo_folder (str): Path to the VALDO folder.
    - dry_run (bool): If True, lists the files that would be deleted without deleting them.
    """
    print(f"Scanning for files in: {valdo_folder} that start with '._sub'")
    deleted = False
    for root, dirs, files in os.walk(valdo_folder):
        for filename in files:
            if filename.startswith('._sub'):
                file_path = os.path.join(root, filename)
                print(f"Found: {file_path}")
                if dry_run:
                    print(f"Dry run: would delete {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        print(f"Successfully deleted: {file_path}")
                        deleted = True
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
    if not deleted and not dry_run:
        print("No files starting with '._sub' were found.")

def rename_files(valdo_folder):
    """
    Renames specific image files within the VALDO folder to more concise names.

    Parameters:
    - valdo_folder (str): Path to the VALDO folder.
    """
    print(f"Starting renaming of files in: {valdo_folder}")
    # Define a mapping from original patterns to new names
    rename_mapping = {
        'sub*_space-T1_desc-masked_FLAIR.nii': 'FLAIR.nii.gz',
        'sub*_space-T1_desc-masked_T2.nii': 'T2.nii.gz',

        
        # Add more mappings as needed
    }

    renamed_files = False
    for root, dirs, files in os.walk(valdo_folder):
        for filename in files:
            for pattern, new_name in rename_mapping.items():
                if fnmatch.fnmatch(filename, pattern):
                    original_path = os.path.join(root, filename)
                    new_path = os.path.join(root, new_name)
                    if os.path.exists(new_path):
                        print(f"Cannot rename {original_path} to {new_path}: Target file already exists.")
                    else:
                        try:
                            os.rename(original_path, new_path)
                            print(f"Renamed: {original_path} -> {new_path}")
                            renamed_files = True
                        except Exception as e:
                            print(f"Failed to rename {original_path}. Reason: {e}")
    if not renamed_files:
        print("No files matched the renaming criteria.")

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
@click.option('--data-folder', default='data/VALDO', type=click.Path(exists=True), help='Path to the dataset folder.')
@click.option('--dry-run', is_flag=True, help='List files to be deleted without deleting them.')
def main(data_folder, dry_run):
    """
    Process and normalize dataset images.
    """
    # Delete ._ files in VALDO folder and its subdirectories
    delete_dot_underscore_files(data_folder, dry_run=dry_run)

    # Proceed with renaming only if not a dry run
    if not dry_run:
        # Rename specific image files
        rename_files(data_folder)

        # Process label files (clean segmentation and transform to instance segmentation task)
        label_files = glob(os.path.join(data_folder, '**', '*-seg.nii.gz'), recursive=True)
        for input_file in tqdm(label_files, desc="Processing label files"):
            process_label(input_file)
        print("All label files processed successfully.")

        # Normalize modalities
        modalities = ['T1', 'FLAIR', 'T2']
        for modality in modalities:
            # Adjust modality names to match the renamed files
            files = glob(os.path.join(data_folder, '**', f'{modality}.nii.gz'), recursive=True)
            if files:
                normalize_modality(modality, files)
            else:
                print(f"No files found for modality: {modality}")

        print("All modalities normalized successfully.")

if __name__ == '__main__':
    main()