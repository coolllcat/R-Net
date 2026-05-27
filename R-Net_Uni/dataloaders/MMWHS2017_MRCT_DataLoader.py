import os 
import torch 
import numpy as np 
import SimpleITK as sitk  

from torch.utils.data import Dataset, DataLoader 
import dataloaders.RegistrationTransforms as RT 

from typing import List, Tuple, Dict 


MMWHS2017_MRCT_LABEL_DICT = {
    "Myocardium": [205], 
    "Left Atrium": [420], 
    "Left Ventricle": [500], 
    "Right Atrium": [550], 
    "Right Ventricle": [600], 
    "Pulmonary Artery": [820], 
    "Ascending Aorta": [850]
}


MMWHS2017_MRCT_RGB_MAPPING_DICT = {
    "Myocardium": [252, 110, 180], 
    "Left Atrium": [51, 82, 171], 
    "Left Ventricle": [109, 95, 177], 
    "Right Atrium": [110, 190, 231], 
    "Right Ventricle": [66, 186, 115], 
    "Pulmonary Artery": [255, 46, 23], 
    "Ascending Aorta": [255, 161, 43]
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


class MMWHS2017_MRCT_Dataset(Dataset): 
    def __init__(self, args, is_train: bool, transforms=None) -> None:
        super().__init__() 

        root_dir = args.root_dir  
        sample_names = os.listdir(root_dir) 
        sample_names = remove_intro_files(sample_names) 
        sample_names = sort_dirs(sample_names) 

        self.is_atlas = args.is_atlas 
        self.is_train = is_train 

        self.sample_dirs = [os.path.join(root_dir, sample_name) for sample_name in sample_names]
        self.train_sample_dirs = self.sample_dirs[:-int(len(self.sample_dirs) * args.split_factor)]
        self.valid_test_sample_dirs = self.sample_dirs[-int(len(self.sample_dirs) * args.split_factor):]
        print("MMWHS2017 CT data split: train: {}, valid and test: {}".format(len(self.train_sample_dirs), len(self.valid_test_sample_dirs))) 

        self.transforms = transforms 

    @staticmethod 
    def _load_image_and_label(sample_dir: str): 
        fixed_image_path = os.path.join(sample_dir, "ct_image_padded.nii.gz") 
        fixed_label_path = os.path.join(sample_dir, "ct_label_padded.nii.gz") 
        moving_image_path = os.path.join(sample_dir, "mr_image_padded.nii.gz")
        moving_label_path = os.path.join(sample_dir, "mr_label_padded.nii.gz")
        fixed_image = sitk.GetArrayFromImage(sitk.ReadImage(fixed_image_path))
        fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label_path))
        moving_image = sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path))
        moving_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_path))

        return moving_image, fixed_image, moving_label, fixed_label 

    def __getitem__(self, index): 
        if self.is_train: 
            moving_image, fixed_image, moving_label, fixed_label = self._load_image_and_label(self.train_sample_dirs[index])
        else: 
            moving_image, fixed_image, moving_label, fixed_label = self._load_image_and_label(self.valid_test_sample_dirs[index])

        # transform here 
        if self.transforms != None: 
            moving_image, fixed_image, moving_label, fixed_label = self.transforms(moving_image, fixed_image, moving_label, fixed_label)
        
        return moving_image, fixed_image, moving_label, fixed_label 
    
    def __len__(self): 
        if self.is_train: 
            return len(self.train_sample_dirs) 
        else: 
            return len(self.valid_test_sample_dirs) 
            
    def __str__(self) -> str:
        return "MMWHS2017_MRCT_Dataset"
         

if __name__ == "__main__": 
    import argparse 
    import RegistrationTransforms as RT 
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--root_dir', type=str, default='/data/postgraduate/wmw/UniRegDatasets/MMWHS2017_Heart_MRCT', help='root dir where stores data') 
    parser.add_argument('--is_atlas', action='store_true') 
    parser.add_argument('--split_factor', type=float, default=0.2, help='how many samples will be splited to be validation or test') 
    parser.add_argument('--batch_size', type=int, default=1, help='batch size') 
    args = parser.parse_args() 

    # set your transforms here 
    train_transforms = RT.Compose([
        RT.AdjustNumpyType(),
        RT.RandomFlip(), 
        RT.Normalize(), 
        RT.AdjustChannels(MMWHS2017_MRCT_LABEL_DICT),   
        RT.ToTensor() 
    ]) 
    valid_test_transforms = RT.Compose([
        RT.AdjustNumpyType(),
        RT.Normalize(), 
        RT.AdjustChannels(MMWHS2017_MRCT_LABEL_DICT), 
        RT.ToTensor() 
    ]) 

    train_dataset = MMWHS2017_MRCT_Dataset(args, True, train_transforms) 
    valid_test_dataset = MMWHS2017_MRCT_Dataset(args, False, valid_test_transforms) 

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4) 
    valid_test_dataloader = DataLoader(valid_test_dataset, batch_size=1, shuffle=False, num_workers=4) 

    print(valid_test_dataset[0][0].shape)
    print(valid_test_dataset[0][1].shape)
    print(valid_test_dataset[0][2].shape)
    print(valid_test_dataset[0][3].shape)
