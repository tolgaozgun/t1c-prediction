import os
import numpy as np
import tensorflow as tf
import nibabel as nib
from sklearn.model_selection import train_test_split

class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset_folder, batch_size, validation_split=0.2):
        self.dataset_folder = dataset_folder
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.data_paths = self._get_data_paths()
        self.train_paths, self.val_paths = train_test_split(self.data_paths, test_size=validation_split, random_state=42)

    def _get_data_paths(self):
        data_paths = []
        sub_folders = sorted(os.listdir(os.path.join(self.dataset_folder, "sourcedata")))
        for sub_folder in sub_folders:
            if sub_folder.startswith("sub-"):
                sub_folder_path = os.path.join(self.dataset_folder, "sourcedata", sub_folder)
                t1c_path = os.path.join(sub_folder_path, f"{os.path.basename(sub_folder_path)}_ce-GADOLINIUM_T1w.nii.gz")
                if os.path.isdir(sub_folder_path) and os.path.exist(t1c_path):
                    data_paths.append(sub_folder_path)
        return data_paths

    def __len__(self):
        return len(self.train_paths) // self.batch_size

    def __getitem__(self, index):
        batch_paths = self.train_paths[index*self.batch_size : (index+1)*self.batch_size]
        batch_x, batch_y = [], []

        for data_path in batch_paths:
            # Load mandatory files
            t1w_path = os.path.join(data_path, f"{os.path.basename(data_path)}_T1w.nii.gz")
            flair_path = os.path.join(data_path, f"{os.path.basename(data_path)}_FLAIR.nii.gz")
            t2w_path = os.path.join(data_path, f"{os.path.basename(data_path)}_T2w.nii.gz")
            t1w_img = nib.load(t1w_path).get_fdata()
            flair_img = nib.load(flair_path).get_fdata()
            t2w_img = nib.load(t2w_path).get_fdata()

            # Load optional files if available
            gadolinium_t1w_path = os.path.join(data_path, f"{os.path.basename(data_path)}_ce-GADOLINIUM_T1w.nii.gz")
            # if os.path.exists(gadolinium_t1w_path): 
            # This check is done in data loading
            gadolinium_t1w_img = nib.load(gadolinium_t1w_path).get_fdata()
                # Add code here to process gadolinium_t1w_img if needed

            # Preprocess and normalize the images as needed
            # Add code here for preprocessing and normalization

            # Append the images to the batch
            batch_x.append([t1w_img, t2w_img, flair_img])  # Input: T1w, T2w, FLAIR
            batch_y.append(gadolinium_t1w_img)  # Target: T1c

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    def get_validation_data(self):
        val_x, val_y = [], []

        for data_path in self.val_paths:
            # Load mandatory files (similar to __getitem__)
            t1w_path = os.path.join(data_path, f"{os.path.basename(data_path)}_T1w.nii.gz")
            flair_path = os.path.join(data_path, f"{os.path.basename(data_path)}_FLAIR.nii.gz")
            t2w_path = os.path.join(data_path, f"{os.path.basename(data_path)}_T2w.nii.gz")
            t1w_img = nib.load(t1w_path).get_fdata()
            flair_img = nib.load(flair_path).get_fdata()
            t2w_img = nib.load(t2w_path).get_fdata()

            # Load optional files if available (similar to __getitem__)
            gadolinium_t1w_path = os.path.join(data_path, f"{os.path.basename(data_path)}_ce-GADOLINIUM_T1w.nii.gz")
            # if not os.path.exists(gadolinium_t1w_path):
            # This test is done in data loading

            gadolinium_t1w_img = nib.load(gadolinium_t1w_path).get_fdata()
                # Add code here to process gadolinium_t1w_img if needed

            # Preprocess and normalize the images as needed (similar to __getitem__)
            # Add code here for preprocessing and normalization

            # Append the images to the validation data
            val_x.append([t1w_img, t2w_img, flair_img])  # Input: T1w, T2w, FLAIR
            val_y.append(gadolinium_t1w_img)  # Target: T1c

        val_x = np.array(val_x)
        val_y = np.array(val_y)
        return val_x, val_y
