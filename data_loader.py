import os
import numpy as np
import tensorflow as tf
import nibabel as nib
from sklearn.model_selection import train_test_split

debug = False

class GaziBrainsDataLoader(tf.keras.utils.Sequence):
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

                # Folder structure: sourcedata/sub-x/anat/*.nii.gz
                # We need to save sub-x as we will need number x later
                # sub_folder_path contains a link to sourcedata/sub-x
                # t1c_folder contains a link to sourcedata/sub-x/anat
                sub_folder_path = os.path.join(self.dataset_folder, "sourcedata", sub_folder)
                t1c_folder = os.path.join(sub_folder_path, "anat")
                
                # Check if T1-C file exists, if not we cannot use this for training
                t1c_path = os.path.join(t1c_folder, f"{os.path.basename(sub_folder_path)}_ce-GADOLINIUM_T1w.nii.gz")
                if os.path.isdir(sub_folder_path) and os.path.exists(t1c_path):
                    data_paths.append(sub_folder_path)
        return data_paths

    def __len__(self):
        return len(self.train_paths) // self.batch_size
    

    def __extract_files__(self, data_path):
        folder_path = os.path.join(data_path, "anat")
        # Load mandatory files
        t1w_path = os.path.join(folder_path, f"{os.path.basename(data_path)}_T1w.nii.gz")
        flair_path = os.path.join(folder_path, f"{os.path.basename(data_path)}_FLAIR.nii.gz")
        t2w_path = os.path.join(folder_path, f"{os.path.basename(data_path)}_T2w.nii.gz")
        t1w_img = nib.load(t1w_path).get_fdata()
        flair_img = nib.load(flair_path).get_fdata()
        t2w_img = nib.load(t2w_path).get_fdata()

        # Load optional files if available
        gadolinium_t1w_path = os.path.join(folder_path, f"{os.path.basename(data_path)}_ce-GADOLINIUM_T1w.nii.gz")
        gadolinium_t1w_img = nib.load(gadolinium_t1w_path).get_fdata()

        # Preprocess and normalize the images as needed
        # Add code here for preprocessing and normalization

        return t1w_img, t2w_img, flair_img, gadolinium_t1w_img


    def __getitem__(self, index) -> tuple[list, list]:
        batch_paths = self.train_paths[index*self.batch_size : (index+1)*self.batch_size]
        batch_x, batch_y = [], []

        for data_path in batch_paths:
            t1w_imgs, t2w_imgs, flair_imgs, gadolinium_t1w_imgs = self.__extract_files__(data_path)


            if debug: 
                print(f't1w_img shape: {t1w_imgs.shape}')
                print(f't2w_img shape: {t2w_imgs.shape}')
                print(f'flair_img shape: {flair_imgs.shape}')
                print(f'gadolinium_t1w_img shape: {gadolinium_t1w_imgs.shape}')

            assert(t1w_imgs.shape[2] == t2w_imgs.shape[2] == flair_imgs.shape[2] == gadolinium_t1w_imgs.shape[2])

            no_of_sequences = t1w_imgs.shape[2]

            for i in range(0, no_of_sequences):
                t1w_img = t1w_imgs[..., i]
                t2w_img = t2w_imgs[..., i]
                flair_img = flair_imgs[..., i]
                gadolinium_t1w_img = gadolinium_t1w_imgs[..., i]

                if debug: 
                    print(f't1w_img shape: {t1w_img.shape}')
                    print(f't2w_img shape: {t2w_img.shape}')
                    print(f'flair_img shape: {flair_img.shape}')
                    print(f'gadolinium_t1w_img shape: {gadolinium_t1w_img.shape}')

                assert(t1w_img.shape == t2w_img.shape == flair_img.shape == gadolinium_t1w_img.shape)

                concatenated_img = np.concatenate([t1w_img[..., np.newaxis], t2w_img[..., np.newaxis], flair_img[..., np.newaxis]], axis=-1)
                batch_x.append(concatenated_img)
                batch_y.append(gadolinium_t1w_img)
                
                if debug: 
                    print(f'Length of batch_x: {len(batch_x)}')
                    print(f'Length of batch_y: {len(batch_y)}')

        return batch_x, batch_y

    def get_train_data(self) -> tuple[list, list]:
        train_x, train_y = [], []

        for data_path in self.train_paths:
            t1w_imgs, t2w_imgs, flair_imgs, gadolinium_t1w_imgs = self.__extract_files__(data_path)


            if debug: 
                print(f't1w_img shape: {t1w_imgs.shape}')
                print(f't2w_img shape: {t2w_imgs.shape}')
                print(f'flair_img shape: {flair_imgs.shape}')
                print(f'gadolinium_t1w_img shape: {gadolinium_t1w_imgs.shape}')

            assert(t1w_imgs.shape[2] == t2w_imgs.shape[2] == flair_imgs.shape[2] == gadolinium_t1w_imgs.shape[2])

            no_of_sequences = t1w_imgs.shape[2]

            for i in range(0, no_of_sequences):
                t1w_img = t1w_imgs[..., i]
                t2w_img = t2w_imgs[..., i]
                flair_img = flair_imgs[..., i]
                gadolinium_t1w_img = gadolinium_t1w_imgs[..., i]

                if debug: 
                    print(f't1w_img shape: {t1w_img.shape}')
                    print(f't2w_img shape: {t2w_img.shape}')
                    print(f'flair_img shape: {flair_img.shape}')
                    print(f'gadolinium_t1w_img shape: {gadolinium_t1w_img.shape}')

                assert(t1w_img.shape == t2w_img.shape == flair_img.shape == gadolinium_t1w_img.shape)

                concatenated_img = np.concatenate([t1w_img[..., np.newaxis], t2w_img[..., np.newaxis], flair_img[..., np.newaxis]], axis=-1)
                train_x.append(concatenated_img)
                train_y.append(gadolinium_t1w_img)

                if debug: 
                    print(f'Length of train_x: {len(train_x)}')
                    print(f'Length of train_y: {len(train_y)}')
            
        return train_x, train_y

    def get_validation_data(self) -> tuple[list, list]:
        val_x, val_y = [], []

        for data_path in self.val_paths:
            t1w_imgs, t2w_imgs, flair_imgs, gadolinium_t1w_imgs = self.__extract_files__(data_path)

            if debug: 
                print(f't1w_img shape: {t1w_imgs.shape}')
                print(f't2w_img shape: {t2w_imgs.shape}')
                print(f'flair_img shape: {flair_imgs.shape}')
                print(f'gadolinium_t1w_img shape: {gadolinium_t1w_imgs.shape}')

            assert(t1w_imgs.shape[2] == t2w_imgs.shape[2] == flair_imgs.shape[2] == gadolinium_t1w_imgs.shape[2])

            no_of_sequences = t1w_imgs.shape[2]

            for i in range(0, no_of_sequences):
                t1w_img = t1w_imgs[..., i]
                t2w_img = t2w_imgs[..., i]
                flair_img = flair_imgs[..., i]
                gadolinium_t1w_img = gadolinium_t1w_imgs[..., i]

                if debug: 
                    print(f't1w_img shape: {t1w_img.shape}')
                    print(f't2w_img shape: {t2w_img.shape}')
                    print(f'flair_img shape: {flair_img.shape}')
                    print(f'gadolinium_t1w_img shape: {gadolinium_t1w_img.shape}')

                assert(t1w_img.shape == t2w_img.shape == flair_img.shape == gadolinium_t1w_img.shape)

                concatenated_img = np.concatenate([t1w_img[..., np.newaxis], t2w_img[..., np.newaxis], flair_img[..., np.newaxis]], axis=-1)
                val_x.append(concatenated_img)
                val_y.append(gadolinium_t1w_img)
                
                if debug: 
                    print(f'Length of val_x: {len(val_x)}')
                    print(f'Length of val_y: {len(val_y)}')

        return val_x, val_y
