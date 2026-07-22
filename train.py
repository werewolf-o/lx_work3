import os, argparse, time, datetime, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import KLDivLoss
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util2.MY_dataset2 import MY_dataset
from util2.augmentation import RandomFlip, RandomCrop
from util2.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from util2.lr_policy import WarmUpPolyLR
from util2.init_func import group_weight
from config import config
from work_03.shiyan_model.Net_09 import Teacher
from Train.lovasz_losses import lovasz_softmax
from work_03.Train.edge_aware_loss import EdgeAwareLoss
#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='Net_14') ####
parser.add_argument('--batch_size', '-b', type=int, default=4)
parser.add_argument('--lr_start', '-ls', type=float, default=6e-5)
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=300)
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=6)
parser.add_argument('--n_class', '-nc', type=int, default=3)
parser.add_argument('--data_dir', '-dr', type=str, default='/media/user/shuju/LXNet/NPO++')
args = parser.parse_args()

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]


def _ice_loss(alpha, y_onehot):
    """
    alpha:    (B, C, H, W)
    y_onehot: (B, C, H, W)  one-hot 标签
    """
    S = alpha.sum(dim=1, keepdim=True)  # (B,1,H,W)
    # digamma 差值：ψ(S) - ψ(α_c)
    loss = y_onehot * (torch.digamma(S) - torch.digamma(alpha))
    return loss.sum(dim=1).mean()  # scalar

Edge = EdgeAwareLoss()

def train(epo, model, train_loader, optimizer):
    model.train()
    loss_seg_sum = 0
    # torch.autograd.set_detect_anomaly(True)
    for it, (images, depth_e, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        depth_e = Variable(depth_e).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)

        start_t = time.time()
        optimizer.zero_grad()

        y_onehot = F.one_hot(labels.long(), num_classes=3).permute(0, 3, 1, 2).float()

        out, out1, out2, out_d,alpha_V,alpha_D,alpha_fused = model(images,depth_e)
        loss1 = F.cross_entropy(out, labels) + lovasz_softmax(F.softmax(out,dim=1), labels)
        loss2 = F.cross_entropy(out1, labels) + lovasz_softmax(F.softmax(out1,dim=1), labels)
        loss3 = F.cross_entropy(out2, labels) + lovasz_softmax(F.softmax(out2,dim=1), labels)
        loss4 = F.cross_entropy(out_d, labels) + lovasz_softmax(F.softmax(out_d, dim=1), labels)

        loss5 = _ice_loss(alpha_V, y_onehot) + _ice_loss(alpha_D, y_onehot) + _ice_loss(alpha_fused, y_onehot)


        loss = loss1 + loss2 + loss3 + loss4 + loss5

        loss.backward()
        optimizer.step()

        loss_seg_sum = loss_seg_sum + loss

        current_idx = (epo - 0) * config.niters_per_epoch + it
        lr = lr_policy.get_lr(current_idx)

        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

        lr_this_epo = 0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        current_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
        print(
            ' %s | Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f' \
            % (current_time, args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr_this_epo,
               len(names) / (time.time() - start_t),
               float(loss),
               ))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
        view_figure = True
        if accIter['train'] % 1000 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:, :3], nrow=8,
                                                    padding=10)
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1,255 // args.n_class)
                groundtruth_tensor = labels.unsqueeze(1) * scale
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor),1)
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = out.argmax(1).unsqueeze(1) * scale
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1)
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
        accIter['train'] = accIter['train'] + 1


