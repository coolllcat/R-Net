import os 
import argparse

from dataloaders.ACDCHeart_MR_DataLoader import ACDCHeart_MR_Dataset, ACDCHeart_MR_LABEL_DICT, ACDCHeart_MR_RGB_MAPPING_DICT
from dataloaders.CHAOS_CT_DataLoader import CHAOS_CT_Dataset, CHAOS_CT_LABEL_DICT, CHAOS_CT_RGB_MAPPING_DICT
from dataloaders.CHAOS_MR_DataLoader import CHAOS_MR_Dataset, CHAOS_MR_LABEL_DICT, CHAOS_MR_RGB_MAPPING_DICT 
from dataloaders.IXI_MR_DataLoader import IXI_MR_Dataset, IXI_MR_LABEL_DICT, IXI_MR_RGB_MAPPING_DICT
from dataloaders.L2RAbdomen_CT_DataLoader import L2RAbdomen_CT_Dataset, L2RAbdomen_CT_LABEL_DICT, L2RAbdomen_CT_RGB_MAPPING_DICT
from dataloaders.L2RAbdomen_MRCT_DataLoader import L2RAbdomen_MRCT_Dataset, L2RAbdomen_MRCT_LABEL_DICT, L2RAbdomen_MRCT_RGB_MAPPING_DICT
from dataloaders.L2RLung_CT_DataLoader import L2RLung_CT_Dataset, L2RLung_CT_LABEL_DICT, L2RLung_CT_RGB_MAPPING_DICT
from dataloaders.L2RNLSTLung_CT_DataLoader import L2RNLSTLung_CT_Dataset, L2RNLSTLung_CT_LABEL_DICT, L2RNLSTLung_CT_RGB_MAPPING_DICT
from dataloaders.LiTS_CT_DataLoader import LiTS_CT_Dataset, LiTS_CT_LABEL_DICT, LiTS_CT_RGB_MAPPING_DICT
from dataloaders.LOLA11Lung_CT_DataLoader import LOLA11Lung_CT_Dataset, LOLA11Lung_CT_LABEL_DICT, LOLA11Lung_CT_RGB_MAPPING_DICT
from dataloaders.LPBA40_MR1_DataLoader import LPBA40_MR1_Dataset, LPBA40_MR1_LABEL_DICT, LPBA40_MR1_RGB_MAPPING_DICT
from dataloaders.MMWHS2017_MRCT_DataLoader import MMWHS2017_MRCT_Dataset, MMWHS2017_MRCT_LABEL_DICT, MMWHS2017_MRCT_RGB_MAPPING_DICT
from dataloaders.MRBrains13_MR_DataLoader import MRBrains13_MR_Dataset, MRBrains13_MR_LABEL_DICT, MRBrains13_MR_RGB_MAPPING_DICT
from dataloaders.MRBrains18_MR_DataLoader import MRBrains18_MR_Dataset, MRBrains18_MR_LABEL_DICT, MRBrains18_MR_RGB_MAPPING_DICT
from dataloaders.MSDLung_CT_DataLoader import MSDLung_CT_Dataset, MSDLung_CT_LABEL_DICT, MSDLung_CT_RGB_MAPPING_DICT
from dataloaders.MSDProstate_MR_DataLoader import MSDProstate_MR_Dataset, MSDProstate_MR_LABEL_DICT, MSDProstate_MR_RGB_MAPPING_DICT
from dataloaders.NIREP_MR_DataLoader import NIREP_MR_Dataset, NIREP_MR_LABEL_DICT, NIREP_MR_RGB_MAPPING_DICT
from dataloaders.NSCLCLung_CT_DataLoader import NSCLCLung_CT_Dataset, NSCLCLung_CT_LABEL_DICT, NSCLCLung_CT_RGB_MAPPING_DICT
from dataloaders.OASIS_MR_DataLoader import OASIS_MR_Dataset, OASIS_MR_LABEL_DICT, OASIS_MR_RGB_MAPPING_DICT
from dataloaders.TCIAPro_MRUS_DataLoader import TCIAPro_MRUS_Dataset, TCIAPro_MRUS_LABEL_DICT, TCIAPro_MRUS_RGB_MAPPING_DICT
from dataloaders.URegPro_MRUS_DataLoader import URegPro_MRUS_Dataset, URegPro_MRUS_LABEL_DICT, URegPro_MRUS_RGB_MAPPING_DICT
from dataloaders.VERSESpine_CT_DataLoader import VERSESpine_CT_Dataset, VERSESpine_CT_LABEL_DICT, VERSESpine_CT_RGB_MAPPING_DICT
from dataloaders.VESSELLung_CT_DataLoader import VESSELLung_CT_Dataset, VESSELLung_CT_LABEL_DICT, VESSELLung_CT_RGB_MAPPING_DICT


import dataloaders.RegistrationTransforms as RT 

from torch.utils.data import DataLoader 


# volume's shape in different dataset 
DATA_SHAPE_MAPPING = {
    'acdc_mr': (128, 224, 192),
    'chaos_ct': (160, 192, 192),
    'chaos_mr': (160, 192, 192),
    'ixi_mr': (160, 192, 224),
    'l2rabdomen_ct': (224, 160, 192),
    'l2rabdomen_mrct': (160, 160, 192),
    'l2rlung_ct': (160, 192, 192),
    'l2rnlstlung_ct': (192, 160, 192),
    'lits_ct': (160, 160, 160),
    'lola11_ct': (160, 160, 160),
    'lpba40_mr1': (160, 192, 160),
    'mmwhs2017_mrct': (96, 96, 96),
    'mrbrains13_mr': (128, 192, 192),
    'mrbrains18_mr': (160, 192, 192),
    'msdlung_ct': (160, 160, 160),
    'msdprostate_mr': (96, 192, 192),
    'nirep_mr': (128, 128, 128),
    'nsclc_ct': (192, 160, 192),
    'oasis_mr': (192, 224, 160),
    'tciapro_mrus': (96, 128, 128),
    'uregpro_mrus': (128, 128, 128),
    'versespine_ct': (160, 128, 128),
    'vessel_ct': (160, 160, 160),
}

# map different label dict 
LABEL_DICT_MAPPING = {
    'acdc_mr': ACDCHeart_MR_LABEL_DICT,
    'chaos_ct': CHAOS_CT_LABEL_DICT, 
    'chaos_mr': CHAOS_MR_LABEL_DICT,
    'ixi_mr': IXI_MR_LABEL_DICT,
    'l2rabdomen_ct': L2RAbdomen_CT_LABEL_DICT,
    'l2rabdomen_mrct': L2RAbdomen_MRCT_LABEL_DICT,
    'l2rlung_ct': L2RLung_CT_LABEL_DICT,
    'l2rnlstlung_ct': L2RNLSTLung_CT_LABEL_DICT,
    'lits_ct': LiTS_CT_LABEL_DICT,
    'lola11_ct': LOLA11Lung_CT_LABEL_DICT,
    'lpba40_mr1': LPBA40_MR1_LABEL_DICT,
    'mmwhs2017_mrct': MMWHS2017_MRCT_LABEL_DICT,
    'mrbrains13_mr': MRBrains13_MR_LABEL_DICT,
    'mrbrains18_mr': MRBrains18_MR_LABEL_DICT,
    'msdlung_ct': MSDLung_CT_LABEL_DICT,
    'msdprostate_mr': MSDProstate_MR_LABEL_DICT,
    'nirep_mr': NIREP_MR_LABEL_DICT,
    'nsclc_ct': NSCLCLung_CT_LABEL_DICT,
    'oasis_mr': OASIS_MR_LABEL_DICT,
    'tciapro_mrus': TCIAPro_MRUS_LABEL_DICT,
    'uregpro_mrus': URegPro_MRUS_LABEL_DICT,
    'versespine_ct': VERSESpine_CT_LABEL_DICT,
    'vessel_ct': VESSELLung_CT_LABEL_DICT,
} 

# dataset sub dir name to find data 
DIR_NAME_MAPPING = {
    'acdc_mr': 'ACDC_Heart_MR',
    'chaos_ct': 'CHAOS_Abdominal_CT', 
    'chaos_mr': 'CHAOS_Abdominal_MR',
    'ixi_mr': 'IXI_Brain_MR_Selected',
    'l2rabdomen_ct': 'Learn2Reg_Abdomen_CT',
    'l2rabdomen_mrct': 'Learn2Reg_Abdomen_MRCT',
    'l2rlung_ct': 'Learn2Reg_Lung_CT',
    'l2rnlstlung_ct': 'Learn2Reg_NLSTLung_CT',
    'lits_ct': 'LiTS_Liver_CT',
    'lola11_ct': 'LOLA11_Lung_CT',
    'lpba40_mr1': 'LPBA40_Brain_MR1',
    'mmwhs2017_mrct': 'MMWHS2017_Heart_MRCT',
    'mrbrains13_mr': 'MRBrains13_Brain_MR',
    'mrbrains18_mr': 'MRBrains18_Brain_MR',
    'msdlung_ct': 'MSD_Lung_CT',
    'msdprostate_mr': 'MSD_Prostate_MR',
    'nirep_mr': 'NIREP_Brain_MR',
    'nsclc_ct': 'NSCLC_Lung_CT',
    'oasis_mr': 'OASIS_Brain_MR',
    'tciapro_mrus': 'TCIA_Prostate_MRUS',
    'uregpro_mrus': 'URegPro_Prostate_MRUS',
    'versespine_ct': 'VerSe_Spine_CT',
    'vessel_ct': 'VESSEL_Lung_CT',
}

# instantize dataset 
DATASET_MAPPING = {
    'acdc_mr': ACDCHeart_MR_Dataset,
    'chaos_ct': CHAOS_CT_Dataset, 
    'chaos_mr': CHAOS_MR_Dataset,
    'ixi_mr': IXI_MR_Dataset,
    'l2rabdomen_ct': L2RAbdomen_CT_Dataset,
    'l2rabdomen_mrct': L2RAbdomen_MRCT_Dataset,
    'l2rlung_ct': L2RLung_CT_Dataset,
    'l2rnlstlung_ct': L2RNLSTLung_CT_Dataset,
    'lits_ct': LiTS_CT_Dataset,
    'lola11_ct': LOLA11Lung_CT_Dataset,
    'lpba40_mr1': LPBA40_MR1_Dataset,
    'mmwhs2017_mrct': MMWHS2017_MRCT_Dataset,
    'mrbrains13_mr': MRBrains13_MR_Dataset,
    'mrbrains18_mr': MRBrains18_MR_Dataset,
    'msdlung_ct': MSDLung_CT_Dataset,
    'msdprostate_mr': MSDProstate_MR_Dataset,
    'nirep_mr': NIREP_MR_Dataset,
    'nsclc_ct': NSCLCLung_CT_Dataset,
    'oasis_mr': OASIS_MR_Dataset,
    'tciapro_mrus': TCIAPro_MRUS_Dataset,
    'uregpro_mrus': URegPro_MRUS_Dataset,
    'versespine_ct': VERSESpine_CT_Dataset,
    'vessel_ct': VESSELLung_CT_Dataset,
} 

# rgb mapping dict 
RGB_MAPPING_DICT = {
    'acdc_mr': ACDCHeart_MR_RGB_MAPPING_DICT,
    'chaos_ct': CHAOS_CT_RGB_MAPPING_DICT, 
    'chaos_mr': CHAOS_MR_RGB_MAPPING_DICT,
    'ixi_mr': IXI_MR_RGB_MAPPING_DICT,
    'l2rabdomen_ct': L2RAbdomen_CT_RGB_MAPPING_DICT,
    'l2rabdomen_mrct': L2RAbdomen_MRCT_RGB_MAPPING_DICT,
    'l2rlung_ct': L2RLung_CT_RGB_MAPPING_DICT,
    'l2rnlstlung_ct': L2RNLSTLung_CT_RGB_MAPPING_DICT,
    'lits_ct': LiTS_CT_RGB_MAPPING_DICT,
    'lola11_ct': LOLA11Lung_CT_RGB_MAPPING_DICT,
    'lpba40_mr1': LPBA40_MR1_RGB_MAPPING_DICT,
    'mmwhs2017_mrct': MMWHS2017_MRCT_RGB_MAPPING_DICT,
    'mrbrains13_mr': MRBrains13_MR_RGB_MAPPING_DICT,
    'mrbrains18_mr': MRBrains18_MR_RGB_MAPPING_DICT,
    'msdlung_ct': MSDLung_CT_RGB_MAPPING_DICT,
    'msdprostate_mr': MSDProstate_MR_RGB_MAPPING_DICT,
    'nirep_mr': NIREP_MR_RGB_MAPPING_DICT,
    'nsclc_ct': NSCLCLung_CT_RGB_MAPPING_DICT,
    'oasis_mr': OASIS_MR_RGB_MAPPING_DICT,
    'tciapro_mrus': TCIAPro_MRUS_RGB_MAPPING_DICT,
    'uregpro_mrus': URegPro_MRUS_RGB_MAPPING_DICT,
    'versespine_ct': VERSESpine_CT_RGB_MAPPING_DICT,
    'vessel_ct': VESSELLung_CT_RGB_MAPPING_DICT,
} 


def add_dataloader_related_args(parser: argparse.ArgumentParser): 
    # add dataset related arguments 
    parser.add_argument('--batch_size', type=int, default=1, help='the number of samples in a batch') 
    parser.add_argument('--dataroot', type=str, default='../', help='root dir where stores data') 
    parser.add_argument('--root_dir', type=str, default="", help='specified dir to store data')
    parser.add_argument('--which_set', type=str, default='', help='used dataset: see __init__.py in dataloaders package for more details')
    # these two arguments are used to control if you need to resize data to avoid memory overflow 
    parser.add_argument('--is_resize', action='store_true') 
    parser.add_argument('--is_crop', action='store_true')
    parser.add_argument('--target_shape', type=str, default='128,128,128', help='target shape used in RT TargetResize') 
    return parser 


def get_transforms(args): 
    """ Return an unified data transforms for all datasets. """ 
    channels_trans = None 
    channels_trans = RT.AdjustChannels(LABEL_DICT_MAPPING[args.which_set]) 

    resize_trans = None 
    if args.is_resize: 
        target_shape = tuple([int(x) for x in args.target_shape.split(',')]) 
        resize_trans = RT.TargetResize(target_shape) 

    center_crop_trans = None 
    if args.is_crop: 
        target_shape = tuple([int(x) for x in args.target_shape.split(',')]) 
        center_crop_trans = RT.CentralCrop(target_shape)  

    train_transforms = RT.Compose([
        RT.AdjustNumpyType(),
        RT.RandomFlip(), 
        RT.Normalize(), 
        channels_trans, 
        resize_trans, 
        center_crop_trans,
        RT.ToTensor() 
    ]) 

    valid_test_transforms = RT.Compose([
        RT.AdjustNumpyType(),
        RT.Normalize(), 
        channels_trans, 
        resize_trans, 
        center_crop_trans,
        RT.ToTensor() 
    ]) 

    return train_transforms, valid_test_transforms 

   
# directly use this function to get your dataloader 
def get_dataloader(args): 
    train_trans, valid_test_trans = get_transforms(args) 

    args.root_dir = os.path.join(args.dataroot, DIR_NAME_MAPPING[args.which_set]) 

    train_dataset = DATASET_MAPPING[args.which_set](args, True, train_trans) 
    valid_test_dataset = DATASET_MAPPING[args.which_set](args, False, valid_test_trans) 

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    valid_test_dataloader = DataLoader(valid_test_dataset, batch_size=1, shuffle=False) 

    return train_dataloader, valid_test_dataloader

