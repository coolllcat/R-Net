""" Utilities """
import os
import logging
import shutil
import torch
import numpy as np 
import torch.nn.functional as F 

from typing import List, Tuple, Dict 


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state.state_dict(), filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best_m.pth.tar')
        shutil.copyfile(filename, best_filename)


def warp3d(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor: 
    """ Warp image or label using deformatin field. 
    
    Parameters: 
        img (Tensor): image or label tensor [B, C, D, H, W] float32 
        flow (Tensor): deformation field [B, 3, D, H, W] float32 
        
    Returns: 
        output (Tensor): warped image or label tensor [B, C, D, H, W] float32 """
    B, _, D, H, W = img.shape
    # mesh grid
    xx = torch.arange(0, W).view(1,1,-1).repeat(D,H,1).view(1,D,H,W)
    yy = torch.arange(0, H).view(1,-1,1).repeat(D,1,W).view(1,D,H,W)
    zz = torch.arange(0, D).view(-1,1,1).repeat(1,H,W).view(1,D,H,W)
    grid = torch.cat((xx,yy,zz),0).repeat(B,1,1,1,1).float().to(img.device) # [bs, 3, D, H, W]
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid[:,0] = 2.0*vgrid[:,0]/(W-1)-1.0 # max(W-1,1)
    vgrid[:,1] = 2.0*vgrid[:,1]/(H-1)-1.0 # max(H-1,1)
    vgrid[:,2] = 2.0*vgrid[:,2]/(D-1)-1.0 # max(D-1,1)
    vgrid = vgrid.permute(0,2,3,4,1) # [bs, D, H, W, 3]        
    output = F.grid_sample(img, vgrid, align_corners=True, padding_mode='border')

    return output


def adjust_image(image: np.ndarray) -> np.ndarray: 
    """ Adjust dataloader's image to standard gray-level image. 

    Parameters: 
        image (Array): image from dataloader [D, H, W] float32 

    Returns: 
        image (Array): gray level image to store [D, H, W] uint8 0~255 """

    _min, _max = np.min(image), np.max(image) 
    image = (image - _min) / (_max - _min) 
    image = image * 255 
    image = np.array(image, dtype='uint8') 

    return image 


def adjust_label(label: np.ndarray) -> np.ndarray: 
    """ Adjust dataloader's label to standard segmentation label. 
    Here we use around to deal with some interpolated values. 
    
    Paramters: 
        label (Array): segmentation label during computation [C, D, H, W] float32 
        
    Returns: 
        results (Array): segmentation label for visualizing [D, H, W] int16 """

    C, D, H, W = label.shape 
    results = np.zeros((D, H, W), dtype='int16') 
    for c in range(C): 
        label_c = label[c, ...] 
        label_c = np.around(label_c) 
        mask = np.array(label_c, dtype='bool') 

        old_label_c = (results * ~mask).astype(np.int16) 
        new_label_c = (mask * (c + 1)).astype(np.int16)
    
        results = old_label_c + new_label_c 

    return results


def get_jdet(displacement: torch.Tensor) -> torch.Tensor:   
    """ Get jac determinent. 
    
    Parameters: 
        displacement (Tensor): [B, 3, D, H, W]
    
    Returns: 
        D (Tensor): jdet [B, D-1, H-1, W-1] """
    displacement = displacement.permute(0,2,3,4,1)

    D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])
    D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])
    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*((D_y[...,1]+1)*(D_z[...,2]+1) - D_y[...,2]*D_z[...,1])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    
    D = D1 - D2 + D3
    
    return D


def get_dice(segment, label, label_dict): 
    # type: (torch.Tensor, torch.Tensor, Dict) -> Dict 
    """ Get dice dict. 
    
    Paramters: 
        segment (Tensor): predicted label [B, C, D, H, W] 
        label (Tensor): gt label [B, C, D, H, W] 
        label dict: dict 
    B should be 1 and C should be number of anatomical areas 
        
    Returns: 
        dice dict """
    segment = segment.squeeze(0) # only remove the batch size axis 
    label = label.squeeze(0)
    assert segment.dim() == 4 and label.dim() == 4
    dices = {}
    for idx, k in enumerate(label_dict.keys()):
        top = (segment[idx] * label[idx]).sum()
        bottom = segment[idx].sum() + label[idx].sum()
        dice = ((2 * top) + 0.001) / (bottom + 0.001)
        dices.update({k: dice.item()})
    return dices


def warp_step(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """ Warp image with step flow. 
    Parameters: 
        img (Tensor): [B, C, D, H, W] 
        flow (Tensor): [B, 1, D, H, W] 
    Returns: 
        img (Tensor): [B, C, D, H, W] """

    img_pad = torch.nn.functional.pad(img, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

    unfold_d = img_pad.unfold(2, 3, 1)  #  torch.Size([1, 1, 1, 3, 3, 3])
    unfold_dh = unfold_d.unfold(3, 3, 1)  # 形状: [1, 1, 3, 3, 5, 3, 3]
    unfold_dhw = unfold_dh.unfold(4, 3, 1)  # 形状: [1, 1, 3, 3, 3, 3, 3, 3]
    unfold_img = unfold_dhw.reshape(*img.shape, -1)
    unfold_img = unfold_img[:, :, :, :, :, [4, 10, 12, 13, 14, 16, 22]]

    flow = flow.permute(0, 2, 3, 4, 1).unsqueeze(1)
    output = (unfold_img * flow).sum(dim=-1)

    return output