import argparse
import random

import torch

import numpy as np


def parser_add_argument():
    """
    return a parser added with args required by fit
    """
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    # Training settings
    parser.add_argument('--model', type=str, default='unet', metavar='N',
                        choices=['unet', 'transunet'],
                        help='neural network used in training')

    parser.add_argument('--backbone', type=str, default='resnet',
                        help='employ with backbone (default: resnet)')

    parser.add_argument('--backbone_pretrained', type=bool, default=True,
                        help='pretrained backbone (default: True)')

    parser.add_argument('--backbone_freezed', type=bool, default=False,
                        help='Freeze backbone to extract features only once (default: False)')

    parser.add_argument('--extract_feat', type=bool, default=True,
                        help='Extract Feature Maps of (default: False) NOTE: --backbone_freezed has to be True for this argument to be considered')

    parser.add_argument('--outstride', type=int, default=8,
                        help='network output stride (default: 16)')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')

    parser.add_argument('--dataset', type=str, default='medical', metavar='N',
                        choices=['coco', 'pascal_voc', 'medical'],
                        help='dataset used for training')

    parser.add_argument('--base_dir', type=str, default='/home/temp58/dataset/biyanai/exp3/date3',
                        help='datasets directory (default = /home/temp58/dataset/biyanai/exp3/date3)')

    # training hyperparams
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: auto)')

    parser.add_argument('--num_classes', type=int,
                        default=0, help='output channel of network')

    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')

    parser.add_argument('--img_size', type=int,
                        default=512, help='input size of network')

    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                                   training (default: auto)')

    parser.add_argument('--test-batch-size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                   testing (default: auto)')

    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--sync_bn', type=bool, default=False,
                        help='whether to use sync bn (default: auto)')

    parser.add_argument('--freeze_bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    # optimizer params
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='adam')

    parser.add_argument('--lr', type=float, default=0.015, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')

    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    parser.add_argument('--evaluation_frequency', type=int, default=5,
                        help='Frequency of model evaluation on training dataset (Default: every 5th round)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')

    parser.add_argument('--seed', type=int, default=1337, metavar='S',
                        help='random seed (default: 1)')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()

    return args


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
