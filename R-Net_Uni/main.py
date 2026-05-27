import os 
import argparse

import torch
import torch.nn as nn 
import numpy as np
import SimpleITK as sitk 
import json 

from dataloaders import get_dataloader, LABEL_DICT_MAPPING 
from model import DeepReg, Step_model_full
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
    train_loader_iter, test_loader = get_dataloader(args) 

    # set model 
    model = Step_model_full()
    model.to(device) 
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print('model number of parameters: {:.3f} MB'.format(num_parameters / 1e6))  

    # initialize optimizer and lr scheduler 
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iterations//400, eta_min=1e-4)

    # set output dir 
    if not os.path.exists(args.checkpoints_dir): 
        os.makedirs(args.checkpoints_dir) 
        print("create output dir: {}".format(args.checkpoints_dir))
    else: 
        print("output dir existed: {}".format(args.checkpoints_dir)) 
    
    if args.resume != '': 
        checkpoint_path = os.path.join(args.checkpoints_dir, args.resume) 
        checkpoint = torch.load(checkpoint_path, map_location='cpu') 
        model.load_state_dict(checkpoint) 

    if args.eval: 
        samples_dir = os.path.join(args.checkpoints_dir, 'samples') 
        if not os.path.join(samples_dir): 
            os.makedirs(samples_dir) 
            print("create test samples dir: {}".format(samples_dir)) 
        else: 
            print("test samples dir existed: {}".format(samples_dir)) 
        
        test(args, model, test_loader) 
        return 0 

    logger = utils.get_logger(os.path.join(args.checkpoints_dir, "train.log")) 
    logger.info("Logger is set - training start") 
    logger.info("model size = {:.3f} MB".format(num_parameters / 1e6))

    iter_start = 0 if args.resume == '' else int(args.resume.split('_')[1]) + 1
    best_dice = 0.0 if args.resume == '' else float(args.resume.split('_')[-1].split('.pth')[0])

    # train loop 
    for iter_num in range(iter_start, args.max_iterations): 
        dataset = list(train_loader_iter.keys())[iter_num % len(train_loader_iter)]
        moving_image, fixed_image, moving_label, fixed_label = next(train_loader_iter[dataset])
        moving_image = moving_image.to(device) 
        fixed_image = fixed_image.to(device) 
        moving_label = moving_label.to(device) 
        fixed_label = fixed_label.to(device)

        # image forward here 
        results = model(moving_image, fixed_image) 
        dice_loss = torch.Tensor([0.]).to(device)
        for idx, result in enumerate(results):
            warped_label = utils.warp3d(moving_label, result) 
            dice_loss += losses.Dice(warped_label, fixed_label)
        grad_loss = losses.Gradient(results[-1]) 

        loss = grad_loss * 1.5 + dice_loss 

        # compute training dice
        flow = results[-1]
        warped_label = utils.warp3d(moving_label, flow) 
        dice = losses.Dice(warped_label, fixed_label, True) 
        jacz = (utils.get_jdet(flow) < 0).float().mean() 

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        logger.info('iteration %d : dataset : %s  loss : %f  dice : %f  jacz : %f  lr : %f' % (iter_num, dataset, loss.item(), dice.item(), jacz, lr_scheduler.get_last_lr()[0]))

        # validation
        if iter_num < 10000:
            validation_freq = 1000
        elif iter_num < 20000:
            validation_freq = 500
        elif iter_num < 30000:
            validation_freq = 250
        else:
            validation_freq = 200
        
        if iter_num >= validation_freq and iter_num % validation_freq == 0: 
            # validate 
            val_dice = valid(model, test_loader, device, logger, iter_num, 50) 
            if val_dice > best_dice:
                best_dice = val_dice
                save_best_path = os.path.join(args.checkpoints_dir, 'best_model.pth') 
                torch.save(model.state_dict(), save_best_path) 

            # save latest model 
            save_mode_path = os.path.join(args.checkpoints_dir, 'iter_{}_dice_{:.3f}.pth'.format(iter_num, best_dice))
            torch.save(model.state_dict(), save_mode_path)

            model.train()

        if iter_num == args.max_iterations: 
            logger.info("end training") 
            return 0 
            
        # if iter_num % 400 == 0:
        #     lr_scheduler.step()


@torch.no_grad() 
def valid(model, test_loader, device, logger, iter_num, max_sample): 
    model.eval()
    dice_list = []
    with torch.no_grad():
        for dataset, loader in test_loader.items():
            dices = utils.AverageMeter()
            jaczs = utils.AverageMeter()

            for step, data in enumerate(loader): 
                moving_image, fixed_image, moving_label, fixed_label = data 

                moving_image = moving_image.to(device) 
                fixed_image = fixed_image.to(device) 
                moving_label = moving_label.to(device)
                fixed_label = fixed_label.to(device)
                N = moving_image.size(0) # batch size 

                results = model(moving_image, fixed_image) 
                flow = results[-1]
                warped_label = utils.warp3d(moving_label, flow)

                # compute dice and jdet ratio 
                dice = losses.Dice(warped_label, fixed_label, True) 
                jacz = (utils.get_jdet(flow) < 0).float().mean() 

                dices.update(dice.item(), N)
                jaczs.update(jacz.item(), N)
                
                if step >= max_sample - 1:
                    break 

            logger.info('iteration %d : dataset : %s  dice : %f  jacz : %f' % (iter_num, dataset, dices.avg, jaczs.avg))
            dice_list.append(dices.avg)

    return np.mean(dice_list)


@torch.no_grad() 
def test(args, model, test_loader):  
    for dataset, loader in test_loader.items():
        for step, data in enumerate(loader): 
            moving_image, fixed_image, moving_label, fixed_label = data 

            moving_image = moving_image.cuda()
            fixed_image = fixed_image.cuda()
            moving_label = moving_label.cuda()
            fixed_label = fixed_label.cuda()
            N = moving_image.size(0) # batch size 

            results = model(moving_image, fixed_image)
            flow = results[-1]
            warped_image = utils.warp3d(moving_image, flow) 
            warped_label = utils.warp3d(moving_label, flow) 

            # get jac det 
            jdet = utils.get_jdet(flow) 
            jdet_ratio = (jdet < 0).float().mean() 

            # get dice 
            dice = utils.get_dice(warped_label, fixed_label, LABEL_DICT_MAPPING[dataset])
            print(dice) 
            dice_mean = losses.Dice(warped_label, fixed_label, True) 

            save_dir = os.path.join(args.checkpoints_dir, "samples_{}".format(dataset), "sample_{}".format(step)) 
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
    parser.add_argument('--max_iterations', type=int, default=100000, help='maximum epoch number to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr for weights')
    parser.add_argument('--gpus', type=str,  default='0', help='GPU to use')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay') 
    parser.add_argument('--supervise', action='store_true', default=False) 
    # add save arguments 
    parser.add_argument('--device', type=str, default='cuda') 
    parser.add_argument('--output_dir', type=str, default='') 
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints') 
    parser.add_argument('--resume', type=str, default='') 
    parser.add_argument('--eval', action='store_true', default=False)  

    # add dataset related arguments 
    parser.add_argument('--batch_size', type=int, default=2, help='batch size') 
    parser.add_argument('--dataroot', type=str, default='/root/private_data/UniRegDatasets', help='root dir where stores data') 
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
    args.which_set = args.which_set.split(',')

    exp_name = 'DeepReg' 
    setproctitle(exp_name) 

    main(args) 
