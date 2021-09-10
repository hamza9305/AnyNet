import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn
#from PIL import Image
import cv2
import numpy as np
import torchvision
#.functional as f

from torch.utils.tensorboard import SummaryWriter

writer =  SummaryWriter("runs/test12")
#---------------------------------------------
#import matplotlib.pyplot as plt
#--------------------------------------------

import models.anynet

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/home/haahm/Undistorted/', help='datapath')
parser.add_argument('--epochs', type=int, default=160,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 24)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 24)')
parser.add_argument('--save_path', type=str, default='/home/haahm/Development/projects/AnyNet/resume_path/',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default='/home/haahm/Development/projects/AnyNet/resume_path/checkpoint.tar',
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=10)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--testwidth', type=int, default=1232)
parser.add_argument('--testheight', type=int, default=368)
parser.add_argument('--dividedispby', type=int, default=256)
parser.add_argument('--outdir', type=str, default=None)

args = parser.parse_args()

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls
elif args.datatype == 'other':
    from dataloader import diy_dataset as ls

def main():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        args.datapath,log, args.split_file)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True, testwidth=args.testwidth, testheight=args.testheight, dividedispby=args.dividedispby),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, testwidth=args.testwidth, testheight=args.testheight, dividedispby=args.dividedispby),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    args.test_dir = args.outdir + '/testing_disparity/'
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    args.train_dir = args.outdir + '/training_disparity/'
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
            args.start_epoch = checkpoint['epoch'] # edited by JS
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    cudnn.benchmark = True
    start_full_time = time.time()
    if args.evaluate:
        test(TestImgLoader, model, log)
        return

    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        adjust_learning_rate(optimizer, epoch)

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = args.save_path + '/checkpoint.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

        if epoch % 1 ==0:
            test(TestImgLoader, model, log,epoch,optimizer) #edited

    test(TestImgLoader, model, log,epoch,optimizer)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):
    global args
    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    D1s = [AverageMeter() for _ in range(stages)]




    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        #L1 loss implementation by default:
        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs = model(imgL, imgR)

        if args.with_spn:
            if epoch >= args.start_epoch_for_spn:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)


        outputs = [torch.squeeze(output, 1) for output in outputs]

        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()


        for idx in range(num_out):
            losses[idx].update(loss[idx].item())
        #L1 loss by default end

        #3-pixel error implementation

        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())

        #3-pixel error end


        if batch_idx % args.print_freq:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)

    writer.add_scalar('training 3-pixel_loss stage 4', D1s[3].avg, epoch)  # edited

    #l1 loss to tensorboard
    writer.add_scalar('traing L1_loss stage 1', losses[0].avg,epoch)  # edited
    writer.add_scalar('traing L1_loss stage 2', losses[1].avg,epoch)  # edited
    writer.add_scalar('traing L1_loss stage 3', losses[2].avg,epoch)  # edited
    writer.add_scalar('traing L1_loss stage 4', losses[3].avg,epoch)  # edited
    #3-pixel error to tensorboard
    writer.add_scalar('training 3-pixel_loss stage 1', D1s[0].avg, epoch)  # edited
    writer.add_scalar('training 3-pixel_loss stage 2', D1s[1].avg, epoch)  # edited
    writer.add_scalar('training 3-pixel_loss stage 3', D1s[2].avg, epoch)  # edited
    writer.add_scalar('training 3-pixel_loss stage 4', D1s[3].avg, epoch)  # edited


    disp = outputs[0]  # edited
    disp_cpu = disp.cpu()  # edited
    print("shape of tensor:", disp_cpu.size())

    #disp_cpu_img = disp_cpu[3, :, :]  # edited
    #print("shape of disp_cpu_img:", disp_cpu_img.size())

    #im = torchvision.transforms.ToPILImage(mode='L')(disp_cpu_img)  # .convert("RGB")
    # print("shape of tensor:", disp_cpu_img.size())   #edited
    # #Image.show('Disparity',disp_cpu_img)
    # height, width  = disp_cpu_img.size()
    # a = torch.FloatTensor(1, height, width) #edired
    # a = f.to_pil_image(a) #edited

    out_path = args.train_dir + ("/image_%05d.png" % epoch)
    #im.save('/home/haahm/Output_disparity_images/initial_dataset/training_disparity/image_' + str(epoch)+ '.png')
#    Image._show(im)
    # exit(0)

   # disp_last_stage = disp[-1,:,:]
   # print(disp_last_stage.shape)

   # img = tensor_to_cv2_image(disp_last_stage)
   # print(img.shape)
   # cv2.imwrite(out_path, img)
   # exit(0)
    #cv2.imshow('my image', img)



# need to uncomment from here---------------------------------------------------------------
#     disp = outputs[2]             #edited
#     print("shape of disp:-------", disp.size())
#
#
#
#     disp1 = disp.reshape(outputs[2].shape,1).unsqueeze(dim=0)
#     disp_cpu = disp.cpu()        #edited
#     print("shape of disp_cpu:-------", disp_cpu.size())
#
#     #im = torchvision.transforms.ToPILImage(mode='RGBA')(disp_cpu) #.convert("RGB")
#
#
#
#     disp_cpu_img = disp_cpu[0, :, :]   #edited
#     print("shape of disp_cpu_img:", disp_cpu_img.size())   #edited
#
#     #----------------------------------------------------------------------
#     disp_cpu1 = disp1.cpu()  # edited
#     print("shape of disp_cpu:-------", disp_cpu1.size())
#
#
#     disp_cpu_img1 = disp_cpu1[:, :, 0]  # edited
#     print("shape of disp_cpu_img1:", disp_cpu_img1.size())  # edited
#
#     im = torchvision.transforms.ToPILImage(mode='L')(disp_cpu_img) #.convert("RGB")
#
#
#
#     #a = f.to_pil_image(mode='RGB')(disp_cpu_img) #(disp_cpu_img) #edited
#     #img2 = torchvision.transforms.ToPILImage(mode="RGBA")(disp)
#     Image._show(im)




def test(dataloader, model, log,epoch, optimizer):  #edited
    global args
    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    losses = [AverageMeter() for _ in range(stages)]

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()


        #l1 loss implementation

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs = model(imgL, imgR)

        if args.with_spn:
            if epoch >= args.start_epoch_for_spn:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)

        outputs = [torch.squeeze(output, 1) for output in outputs]

        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        #l1 loss implementation end


        #3 pixel error by default
        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())
        #3 pixel error by default end
        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)


    #l1_loss to tensorboard
    writer.add_scalar('testing L1_loss stage 1', losses[0].avg, epoch)  # edited
    writer.add_scalar('testing L1_loss stage 2', losses[1].avg, epoch)  # edited
    writer.add_scalar('testing L1_loss stage 3', losses[2].avg, epoch)  # edited
    writer.add_scalar('testing L1_loss stage 4', losses[3].avg, epoch)  # edited

    #3-pixel error to tensorbaard
    writer.add_scalar('testing 3-pixel_loss stage 1', D1s[0].avg,epoch)  # edited
    writer.add_scalar('testing 3-pixel_loss stage 2', D1s[1].avg, epoch)  # edited
    writer.add_scalar('testing 3-pixel_loss stage 3', D1s[2].avg, epoch)  # edited
    writer.add_scalar('testing 3-pixel_loss stage 4', D1s[3].avg, epoch)  # edited



#--------------------------finetune kitti on sceneflow------------------

    disp = output[0]  # edited
    print("shape of disp",disp.size())
    disp_cpu = disp.cpu()  # edited
    print("shape of tensor:", disp_cpu.size())

    gt = disp_L[0]
    print(gt.shape)

    #disp_cpu_img = disp_cpu[3, :, :]  # edited
    #print("shape of disp_cpu_img:", disp_cpu_img.size())

    #im = torchvision.transforms.ToPILImage(mode='L')(disp_cpu_img)  # .convert("RGB")
    # print("shape of tensor:", disp_cpu_img.size())   #edited
     
    out_path1 = args.test_dir + ("/image_%05d.png" % epoch)
    out_path2 = args.test_dir + ("/debug_%05d.png" % epoch)
    #im.save(outdir + '/testing_disparity/image_' + str(epoch)+ '.png')
    #Image._show(im)

    disp_last_stage1 = disp
    print(disp_last_stage1.shape)

    img1 = tensor_to_cv2_image(disp_last_stage1)
    gt1 = tensor_to_cv2_image(gt)

    debug_img = np.hstack((img1, gt1))

    print(img1.shape)
    #cv2.imwrite(out_path1, img1)
    cv2.imwrite(out_path2, debug_img)


#---------------finetune kitto on scene flow------------







    # disp = outputs[2]  # edited
    # print("shape of disp:-------", disp.size())
    #
    # disp1 = disp.reshape(outputs[2].shape, 1).unsqueeze(dim=0)
    # disp_cpu = disp.cpu()  # edited
    # print("shape of disp_cpu:-------", disp_cpu.size())
    #
    # # im = torchvision.transforms.ToPILImage(mode='RGBA')(disp_cpu) #.convert("RGB")
    #
    # disp_cpu_img = disp_cpu[0, :, :]  # edited
    # #exit(0)
    # print("shape of disp_cpu_img:", disp_cpu_img.size())  # edited
    #
    # # ----------------------------------------------------------------------
    # disp_cpu1 = disp1.cpu()  # edited
    # #print("shape of disp_cpu:-------", disp_cpu1.size())
    #
    # disp_cpu_img1 = disp_cpu1[:, :, 0]  # edited
    # print("shape of disp_cpu_img1:", disp_cpu_img1.size())  # edited
    #
    # im = torchvision.transforms.ToPILImage(mode='L')(disp_cpu)  # .convert("RGB")
    #
    # # a = f.to_pil_image(mode='RGB')(disp_cpu_img) #(disp_cpu_img) #edited
    # # img2 = torchvision.transforms.ToPILImage(mode="RGBA")(disp)
    # Image._show(im)



def tensor_to_cv2_image(tensor: torch.Tensor):
    """
    converts a PyTorch disparity map s.th. it can be read by OpenCV 2+

    Args:
        tensor: the Pytorch tensor

    Returns:
        A numpy array of the shape (y-dim, x-dim, 3)

    Author:
        Julian Seuffert
    """

    np_image = tensor.data.cpu().numpy()
    print("i am here",np_image.shape)
    if len(np_image.shape) != 3:
        # grayscale image
        np_image = np.expand_dims(np_image, axis=2)
        np_image = np.concatenate([np_image, np_image, np_image], axis=2)
        return np_image
    else:
        # rgb image
        r = np_image[0,:,:]
        g = np_image[1,:,:]
        b = np_image[2,:,:]
        r = np.expand_dims(r, axis=2)
        g = np.expand_dims(g, axis=2)
        b = np.expand_dims(b, axis=2)
        bgr_cv2 = np.concatenate((b,g,r), axis=2)
        return bgr_cv2





def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
