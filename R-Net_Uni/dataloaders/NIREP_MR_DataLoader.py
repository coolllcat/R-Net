import os 
import torch 
import numpy as np 
import SimpleITK as sitk  

from torch.utils.data import Dataset, DataLoader 
import dataloaders.RegistrationTransforms as RT 

from typing import List, Tuple, Dict 


NIREP_MR_LABEL_DICT = {
    'Occipital': [1, 2],  #  枕叶
    'Cingulate': [3, 4],  #  扣带回
    'Insular': [5, 6],  #  岛叶（岛皮质）
    'Temporal_polar': [7, 8],  #  颞极
    'Temporal_lobe': [9, 10, 11, 12],  #  颞叶
    'Hippocampus': [13, 14],  #  海马旁回
    'Prefrontal': [15, 16],  #  前额皮层
    'Frontal': [17, 18, 19, 20],  #  额叶
    'Orbitofrontal': [23, 24],  #  眶额皮层
    'Parietal': [25, 26, 27, 28, 29, 30, 31, 32],  #  顶叶
    'Callosum': [33],  #  胼胝体
}


NIREP_MR_RGB_MAPPING_DICT = {
    'Occipital': [230, 148, 34],
    'Cingulate': [245, 245, 245],
    'Insular': [119, 159, 176],
    'Temporal_polar': [205, 62, 78],
    'Temporal_lobe': [220, 248, 164],
    'Hippocampus': [0, 118, 14],
    'Prefrontal': [236, 13, 176],
    'Frontal': [165, 42, 42],
    'Orbitofrontal': [220, 216, 20],
    'Parietal': [122, 186, 220],
    'Callosum': [120, 18, 134], 
} 


def sort_dirs(names: List[str]) -> List[str]: 
    indices = sorted(
        range(len(names)), 
        key=lambda index: int(names[index].split('_')[1]) 
    ) 
    sorted_names = [names[index] for index in indices] 
    return sorted_names


def remove_intro_files(sample_names: List[str]): 
    name_buffer = [] 
    for sample_name in sample_names: 
        if sample_name.find('.') == -1: 
            name_buffer.append(sample_name) 
    return name_buffer 


class NIREP_MR_Dataset(Dataset): 
    def __init__(self, args, is_train: bool, transforms=None) -> None:
        super().__init__() 

        root_dir = args.root_dir  
        sample_names = os.listdir(root_dir) 
        sample_names = remove_intro_files(sample_names) 
        sample_names = sort_dirs(sample_names) 

        self.is_atlas = args.is_atlas 
        self.is_train = is_train 

        if self.is_atlas: 
            fixed_sample_name = sample_names[0] 
            self.fixed_sample_dir = os.path.join(root_dir, fixed_sample_name) 

            moving_sample_names = sample_names[1:] 
            split_factor = float(args.split_factor) 
            num_moving = len(moving_sample_names) 
            num_valid_test = int(num_moving * split_factor) 
            num_train = num_moving - num_valid_test 
            train_moving_sample_names = moving_sample_names[:-num_valid_test] 
            valid_test_moving_sample_names = moving_sample_names[-num_valid_test:] 
            print("NIREP MR data split: train: {}, valid and test: {}".format(num_train, num_valid_test)) 

            train_sample_dirs = [os.path.join(root_dir, name) for name in train_moving_sample_names] 
            valid_test_sample_dirs = [os.path.join(root_dir, name) for name in valid_test_moving_sample_names] 
            self.train_sample_dirs = train_sample_dirs 
            self.valid_test_sample_dirs = valid_test_sample_dirs 
        else: 
            num_samples = len(sample_names) 
            split_factor = float(args.split_factor) 
            num_valid_test = int(num_samples * split_factor) 
            sample_names_train = sample_names[:-num_valid_test] 
            sample_names_valid_test = sample_names[-num_valid_test:] 

            sample_pairs_train = [] 
            num_samples_train = len(sample_names_train) 
            for i in range(num_samples_train): 
                for j in range(num_samples_train): 
                    if i != j: 
                        sample_dir_i = os.path.join(root_dir, sample_names_train[i]) 
                        sample_dir_j = os.path.join(root_dir, sample_names_train[j]) 
                        pairs = (sample_dir_i, sample_dir_j) 
                        sample_pairs_train.append(pairs) 
            self.sample_pairs_train = sample_pairs_train 

            sample_pairs_valid_test = [] 
            num_samples_valid_test = len(sample_names_valid_test) 
            for i in range(num_samples_valid_test): 
                for j in range(num_samples_valid_test): 
                    if i != j: 
                        sample_dir_i = os.path.join(root_dir, sample_names_valid_test[i]) 
                        sample_dir_j = os.path.join(root_dir, sample_names_valid_test[j]) 
                        pairs = (sample_dir_i, sample_dir_j) 
                        sample_pairs_valid_test.append(pairs) 
            self.sample_pairs_valid_test = sample_pairs_valid_test 
            print("NIREP MR data split: train: {}, valid and test: {}".format(len(self.sample_pairs_train), len(self.sample_pairs_valid_test)))

        self.transforms = transforms 

    @staticmethod 
    def _load_image_and_label(sample_dir: str): 
        image_path = os.path.join(sample_dir, "image.nii.gz") 
        label_path = os.path.join(sample_dir, "label.nii.gz") 
        image, label = sitk.GetArrayFromImage(sitk.ReadImage(image_path)), sitk.GetArrayFromImage(sitk.ReadImage(label_path)) 

        image = image[:, 22:278, :]
        label = label[:, 22:278, :]
        image = image[::2, ::2, ::2]
        label = label[::2, ::2, ::2]
        return image, label  

    def __getitem__(self, index): 
        if self.is_atlas: 
            fixed_image, fixed_label = self._load_image_and_label(self.fixed_sample_dir) 
            if self.is_train: 
                moving_image, moving_label = self._load_image_and_label(self.train_sample_dirs[index]) 
            else: 
                moving_image, moving_label = self._load_image_and_label(self.valid_test_sample_dirs[index]) 
        else: 
            if self.is_train: 
                sample_pair_train = self.sample_pairs_train[index] 
                fixed_sample_dir, moving_sample_dir = sample_pair_train 
            else: 
                sample_pair_valid_test = self.sample_pairs_valid_test[index] 
                fixed_sample_dir, moving_sample_dir = sample_pair_valid_test 
            fixed_image, fixed_label = self._load_image_and_label(fixed_sample_dir) 
            moving_image, moving_label = self._load_image_and_label(moving_sample_dir) 

        # transform here 
        if self.transforms != None: 
            moving_image, fixed_image, moving_label, fixed_label = self.transforms(moving_image, fixed_image, moving_label, fixed_label)
        
        return moving_image, fixed_image, moving_label, fixed_label 
    
    def __len__(self): 
        if self.is_atlas: 
            if self.is_train: 
                return len(self.train_sample_dirs) 
            else: 
                return len(self.valid_test_sample_dirs) 
        else: 
            if self.is_train: 
                return len(self.sample_pairs_train) 
            else: 
                return len(self.sample_pairs_valid_test) 
            
    def __str__(self) -> str:
        return "NIREP_MR_Dataset"
         

if __name__ == "__main__": 
    import argparse 
    import RegistrationTransforms as RT 
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--root_dir', type=str, default='/data/postgraduate/wmw/UniRegDatasets/NIREP_Brain_MR', help='root dir where stores data') 
    parser.add_argument('--is_atlas', action='store_true') 
    parser.add_argument('--split_factor', type=float, default=0.2, help='how many samples will be splited to be validation or test') 
    parser.add_argument('--batch_size', type=int, default=1, help='batch size') 
    args = parser.parse_args() 

    # set your transforms here 
    train_transforms = RT.Compose([
        RT.AdjustNumpyType(),
        RT.RandomFlip(), 
        RT.Normalize(), 
        RT.AdjustChannels(NIREP_MR_LABEL_DICT),  
        RT.ToTensor() 
    ]) 
    valid_test_transforms = RT.Compose([
        RT.AdjustNumpyType(),
        RT.Normalize(), 
        RT.AdjustChannels(NIREP_MR_LABEL_DICT), 
        RT.ToTensor() 
    ]) 

    train_dataset = NIREP_MR_Dataset(args, True, train_transforms) 
    valid_test_dataset = NIREP_MR_Dataset(args, False, valid_test_transforms) 

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4) 
    valid_test_dataloader = DataLoader(valid_test_dataset, batch_size=1, shuffle=False, num_workers=4) 

    print(valid_test_dataset[0][0].shape)
    print(valid_test_dataset[0][1].shape)
    print(valid_test_dataset[0][2].shape)
    print(valid_test_dataset[0][3].shape)
