import autorootcwd
import os
import random
import shutil

def split_dataset(root_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    datasets = ['BRATS_GLI', 'BRATS_MEN', 'BRATS_MET', 'BRATS_PED']

    for dataset in datasets:
        dataset_path = os.path.join(root_dir, dataset)
        train_path = os.path.join(dataset_path, 'Train')
        val_path = os.path.join(dataset_path, 'Val')
        test_path = os.path.join(dataset_path, 'Test')

        # Create Val and Test directories if they don't exist
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Get all subject directories
        subjects = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
        total_subjects = len(subjects)

        # Calculate the number of subjects for each set
        num_val = int(total_subjects * val_ratio)
        num_test = int(total_subjects * test_ratio)
        num_train = total_subjects - num_val - num_test

        # Randomly shuffle the subjects
        random.shuffle(subjects)

        # Split the subjects
        val_subjects = subjects[:num_val]
        test_subjects = subjects[num_val:num_val+num_test]
        train_subjects = subjects[num_val+num_test:]

        # Move subjects to their respective directories
        for subject in val_subjects:
            shutil.move(os.path.join(train_path, subject), os.path.join(val_path, subject))

        for subject in test_subjects:
            shutil.move(os.path.join(train_path, subject), os.path.join(test_path, subject))

        print(f"{dataset} dataset split complete:")
        print(f"Train: {len(train_subjects)}")
        print(f"Val: {len(val_subjects)}")
        print(f"Test: {len(test_subjects)}")
        print()

if __name__ == "__main__":
    root_dir = "data"
    split_dataset(root_dir)
    print("Dataset splitting completed successfully.")
