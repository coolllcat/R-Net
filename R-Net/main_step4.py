import os 
import argparse

import torch
import torch.nn as nn 
import numpy as np
import SimpleITK as sitk 
import json 

from dataloaders import get_dataloader, LABEL_DICT_MAPPING 
from model import Step_model_full, Step_model_distill
import utils 
import losses

from setproctitle import setproctitle 


def main(args): 
    print(args) 

    device = torch.device(args.device) 

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # set dataset and dataloader  
    train_loader, test_loader = get_dataloader(args) 

    # set model 
    teacher_model = Step_model_full(base_dim=args.base_dim, steps=args.steps, step=args.step)
    teacher_model.load_state_dict(torch.load(args.output_dir.replace('step4', 'step3') + '/best_model.pth', map_location='cpu')) 
    model = Step_model_distill(teacher_model)
    model.to(device) 
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print('model number of parameters: {:.3f} MB'.format(num_parameters / 1e6))  

    # initialize optimizer and lr scheduler 
    max_epoch = args.max_iterations // len(train_loader) + 1 
    optimizer = torch.optim.Adam(model.flowout.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iterations//400, eta_min=1e-4)

    # set output dir 
    output_dir = args.output_dir 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
        print("create output dir: {}".format(output_dir))
    else: 
        print("output dir existed: {}".format(output_dir)) 
    
    if args.resume != '': 
        checkpoint_path = os.path.join(output_dir, args.resume) 
        checkpoint = torch.load(checkpoint_path, map_location='cpu') 
        model.load_state_dict(checkpoint) 

    if args.eval: 
        samples_dir = os.path.join(output_dir, 'samples') 
        if not os.path.join(samples_dir): 
            os.makedirs(samples_dir) 
            print("create test samples dir: {}".format(samples_dir)) 
        else: 
            print("test samples dir existed: {}".format(samples_dir)) 
        
        test(args, model, test_loader) 
        return 0 

    logger = utils.get_logger(os.path.join(args.output_dir, "train.log")) 
    logger.info("Logger is set - training start") 
    logger.info("model size = {:.3f} MB".format(num_parameters / 1e6))

    iter_num = 0
    best_dice = 0.0

    # train loop 
    for eopch in range(max_epoch): 
        for step, data in enumerate(train_loader): 

            moving_image, fixed_image, moving_label, fixed_label = data 
            moving_image = moving_image.to(device) 
            fixed_image = fixed_image.to(device) 
            moving_label = moving_label.to(device) 
            fixed_label = fixed_label.to(device)

            # image forward here 
            flow = model(moving_image, fixed_image) 
            warped_image = utils.warp3d(moving_image, flow)
            warped_label = utils.warp3d(moving_label, flow) 
            grad_loss = losses.Gradient(flow) 
            dice_loss = losses.Dice(warped_label, fixed_label)
            ncc_loss = losses.Ncc(warped_image, fixed_image)

            loss = ncc_loss * args.beta + grad_loss * args.lamda + dice_loss 

            # compute training dice
            dice = losses.Dice(warped_label, fixed_label, True) 

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            iter_num += 1 
            logger.info('iteration %d : loss : %f  dice : %f  lr : %f' % (iter_num, loss.item(), dice.item(), lr_scheduler.get_last_lr()[0]))

            # validation
            if iter_num >= 200 and iter_num % 200 == 0: 

                # validate 
                val_dice = valid(model, test_loader, device, logger, iter_num, 50) 
                if val_dice > best_dice:
                    best_dice = val_dice
                    save_best_path = os.path.join(args.output_dir, 'best_model.pth') 
                    torch.save(model.state_dict(), save_best_path) 

                # save latest model 
                save_mode_path = os.path.join(args.output_dir, 'iter_{}_dice_{:.3f}.pth'.format(iter_num, best_dice))
                torch.save(model.state_dict(), save_mode_path)

                model.train()

            if iter_num == args.max_iterations: 
                logger.info("end training") 
                return 0 
                
            # if iter_num % 400 == 0:
            #     lr_scheduler.step()

@torch.no_grad() 
def valid(model, test_loader, device, logger, iter_num, max_sample): 
    dices = utils.AverageMeter()
    jaczs = utils.AverageMeter()
    
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(test_loader): 
            moving_image, fixed_image, moving_label, fixed_label = data 

            moving_image = moving_image.to(device) 
            fixed_image = fixed_image.to(device) 
            moving_label = moving_label.to(device)
            fixed_label = fixed_label.to(device)
            N = moving_image.size(0) # batch size 

            flow = model(moving_image, fixed_image) 
            warped_label = utils.warp3d(moving_label, flow)

            # compute dice and jdet ratio 
            dice = losses.Dice(warped_label, fixed_label, True) 
            jacz = (utils.get_jdet(flow) < 0).float().mean() 

            dices.update(dice.item(), N)
            jaczs.update(jacz.item(), N)
            
            if step >= max_sample - 1:
                break 

    logger.info('iteration %d : dice : %f  jacz : %f' % (iter_num, dices.avg, jaczs.avg))

    return dices.avg

@torch.no_grad() 
def test(args, model, test_loader):  
    for step, data in enumerate(test_loader): 
        moving_image, fixed_image, moving_label, fixed_label = data 

        moving_image = moving_image.cuda()
        fixed_image = fixed_image.cuda()
        moving_label = moving_label.cuda()
        fixed_label = fixed_label.cuda()
        N = moving_image.size(0) # batch size 

        flow = model(moving_image, fixed_image)
        warped_image = utils.warp3d(moving_image, flow) 
        warped_label = utils.warp3d(moving_label, flow) 

        # get jac det 
        jdet = utils.get_jdet(flow) 
        jdet_ratio = (jdet < 0).float().mean() 

        # get dice 
        dice = utils.get_dice(warped_label, fixed_label, LABEL_DICT_MAPPING[args.which_set])
        print(dice) 
        dice_mean = losses.Dice(warped_label, fixed_label, True) 

        save_dir = os.path.join(args.output_dir, "samples", "sample_{}".format(step)) 
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir) 
        
        # save image 
        sitk.WriteImage(sitk.GetImageFromArray(utils.adjust_image(moving_image.squeeze().cpu().numpy())), os.path.join(save_dir, 'moving_image.nii.gz')) 
        sitk.WriteImage(sitk.GetImageFromArray(utils.adjust_image(fixed_image.squeeze().cpu().numpy())), os.path.join(save_dir, 'fixed_image.nii.gz')) 
        sitk.WriteImage(sitk.GetImageFromArray(utils.adjust_image(warped_image.squeeze().cpu().numpy())), os.path.join(save_dir, 'warped_image.nii.gz')) 
        # save label only remove the batch size axis 
        sitk.WriteImage(sitk.GetImageFromArray(utils.adjust_label(moving_label.squeeze(0).cpu().numpy())), os.path.join(save_dir, 'moving_label.nii.gz')) 
        sitk.WriteImage(sitk.GetImageFromArray(utils.adjust_label(fixed_label.squeeze(0).cpu().numpy())), os.path.join(save_dir, 'fixed_label.nii.gz')) 
        sitk.WriteImage(sitk.GetImageFromArray(utils.adjust_label(warped_label.squeeze(0).cpu().numpy())), os.path.join(save_dir, 'warped_label.nii.gz')) 
        # save disp 
        sitk.WriteImage(sitk.GetImageFromArray(flow.squeeze().permute(1, 2, 3, 0).cpu().numpy()), os.path.join(save_dir, 'disp.nii.gz')) 
        # save jdet 
        sitk.WriteImage(sitk.GetImageFromArray(jdet.squeeze().cpu().numpy()), os.path.join(save_dir, 'jdet.nii.gz')) 
        # save dice 
        with open(os.path.join(save_dir, 'dice.json'), "w", encoding='utf-8') as f:
            json.dump(dice, f) 
        f.close() 
        # save jdet ratio 
        with open(os.path.join(save_dir, 'jdet_ratio.json'), 'w', encoding='utf-8') as f: 
            json.dump({'jdet_ratio': jdet_ratio.item()}, f) 
        f.close() 
        with open(os.path.join(save_dir, 'dice_mean.json'), 'w', encoding='utf-8') as f: 
            json.dump({'dice_mean': dice_mean.item()}, f) 
        f.close() 

        if step >= 50:
            break


if __name__ == "__main__": 

    parser = argparse.ArgumentParser() 

    # add train related arguments 
    parser.add_argument('--max_iterations', type=int, default=8000, help='maximum epoch number to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr for weights')
    parser.add_argument('--gpus', type=str,  default='0', help='GPU to use')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay') 
    parser.add_argument('--supervise', action='store_true', default=False) 
    # add save arguments 
    parser.add_argument('--device', type=str, default='cuda') 
    parser.add_argument('--output_dir', type=str, default='') 
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_step4') 
    parser.add_argument('--resume', type=str, default='') 
    parser.add_argument('--eval', action='store_true', default=False)  
    # add model related arguments
    parser.add_argument('--base_dim', type=int, default=64, help='base dim of model')
    parser.add_argument('--steps', type=int, default=15, help='steps of model')
    parser.add_argument('--step', type=int, default=1, help='step of model')
    parser.add_argument('--lamda', type=float, default=0.1, help='weight for gradient loss')
    parser.add_argument('--beta', type=float, default=1, help='weiht for ncc loss')
    # add dataset related arguments 
    parser.add_argument('--batch_size', type=int, default=2, help='batch size') 
    parser.add_argument('--dataroot', type=str, default='/data/postgraduate/wmw/UniRegDatasets', help='root dir where stores data') 
    parser.add_argument('--which_set', type=str, default='chaos_ct', help='which dataset will be used')
    # these two arguments are used to control if you need to resize data to avoid memory overflow 
    parser.add_argument('--is_crop', action='store_true')
    parser.add_argument('--is_resize', action='store_true') 
    parser.add_argument('--target_shape', type=str, default='', help='target shape used in RT TargetResize') 
    # note that we only use split factor to chunk test sub set when atlas mode is used 
    parser.add_argument('--is_atlas', action='store_true')  
    parser.add_argument('--split_factor', type=float, default=0.2, help='how many samples will be splited to be validation or test') 
    args = parser.parse_args() 

    # set exp name 
    exp_name = '' 
    if args.which_set == 'acdc_mr': 
        exp_name += 'ACDC_MR_'
    elif args.which_set == 'chaos_ct': 
        exp_name += 'CHAOS_CT_' 
    elif args.which_set == 'chaos_mr': 
        exp_name += 'CHAOS_MR_' 
    elif args.which_set == 'ixi_mr':
        exp_name += 'IXI_MR_'
    elif args.which_set == 'l2rabdomen_ct':
        exp_name += 'L2RAbdomen_CT_'
    elif args.which_set == 'l2rabdomen_mrct':
        exp_name += 'L2RAbdomen_MRCT_'
    elif args.which_set == 'l2rlung_ct':
        exp_name += 'L2RLung_CT_'
    elif args.which_set == 'l2rnlstlung_ct':
        exp_name += 'L2RNLSTLung_CT_'
    elif args.which_set == 'lits_ct':
        exp_name += 'LITS_CT_'
    elif args.which_set == 'lola11_ct':
        exp_name += 'LOLA11Lung_CT_'
    elif args.which_set == 'lpba40_mr1':
        exp_name += 'LPBA40_MR1_'
    elif args.which_set == 'mmwhs2017_mrct':
        exp_name += 'MMWHS2017_MRCT_'
    elif args.which_set == 'mrbrains13_mr':
        exp_name += 'MRBRAINS13_MR_'
    elif args.which_set == 'mrbrains18_mr':
        exp_name += 'MRBRAINS18_MR_'
    elif args.which_set == 'msdlung_ct':
        exp_name += 'MSDLung_CT_'
    elif args.which_set == 'nirep_mr':
        exp_name += 'NIREP_MR_'
    elif args.which_set == 'nsclc_ct':
        exp_name += 'NSCLCLung_CT_'
    elif args.which_set == 'oasis_mr':
        exp_name += 'OASIS_MR_'
    elif args.which_set == 'tciapro_mrus':
        exp_name += 'TCIAPro_MRUS_'
    elif args.which_set == 'uregpro_mrus':
        exp_name += 'UREGPRO_MRUS_'
    elif args.which_set == 'versespine_ct':
        exp_name += 'VERSESpine_CT_'
    elif args.which_set == 'vessel_ct':
        exp_name += 'VESSELLung_CT_'
    else: 
        raise NotImplementedError() 
    
    if args.is_atlas: 
        exp_name += "A" 
    else: 
        exp_name += "NA" 

    if args.supervise:
        exp_name += "_SUP"
        
    args.output_dir = os.path.join(args.checkpoints_dir, exp_name) 

    setproctitle(exp_name) 

    main(args) 
