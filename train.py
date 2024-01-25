import datetime
import logging
import os
import torch
import numpy as np
from tqdm import tqdm

from config.config import parser_add_argument, set_random_seed
from dataloaders import data_loader
from model import create_model
import torch.nn.functional as F
from medpy import metric

from model.batchnorm_utils import patch_replication_callback
from utils.dice_score import dice_loss
from utils.loss import SegmentationLosses

from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


def main():
    now = datetime.datetime.now()
    time_start = now.strftime("%Y-%m-%d %H:%M:%S")

    logging.info("Executing Image Segmentation at time: {0}".format(time_start))
    # args and seed
    args = parser_add_argument()
    print(args)
    set_random_seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epochs = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'medical': 50,
        }
        args.epochs = epochs[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'medical': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = "{0}_{1}_epo{2}_bs{3}_lr{4}_{5}_s{6}_time{7}".format(
            str(args.model),
            str(args.backbone),
            str(args.epochs),
            str(args.batch_size),
            str(args.lr),
            str(args.img_size),
            str(args.seed),
            str(time_start)
        )
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.directory)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        args.workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.num_classes = data_loader(args, **kwargs)
        args.num_classes = self.num_classes
        print(self.num_classes)

        # create model
        self.model = create_model(args)

        # Define Criterion
        self.criterion = SegmentationLosses().build_loss(mode=args.loss_type)

        # Define Optimizer
        if args.optimizer == "sgd":

            if args.backbone_freezed:
                self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=args.lr * 10,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov,
                )
            else:
                train_params = [
                    {"params": self.model.get_1x_lr_params(), "lr": args.lr},
                    {"params": self.model.get_10x_lr_params(), "lr": args.lr},
                ]

                self.optimizer = torch.optim.SGD(
                    train_params,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov,
                )
        else:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        # Define Evaluator
        self.evaluator = Evaluator(self.num_classes)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image = image.to(dtype=torch.float32)
            target = target.to(dtype=torch.long)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            # loss = self.criterion(output, target)
            loss = self.criterion(output, target).cuda() \
                   + dice_loss(F.softmax(output, dim=1).float(),
                               F.one_hot(target, self.model.module.n_classes).permute(0, 3, 1, 2).float(),
                               multiclass=False)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, filename='checkpoint_epoch%' % epoch)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        dice = 0
        test_total = 0.0
        asd = 0
        count = 0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            target = target.to(dtype=torch.long)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target).cuda()
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            test_total += target.size(0)

            np.set_printoptions(threshold=np.inf)
            pred = output.cpu().numpy()
            target = target.cpu().numpy()

            # asd = metric.binary.asd(pred, target)
            pred = np.argmax(pred, axis=1)
            if pred.sum() == 0:
                asd += 100
            else:
                asd += metric.binary.asd(pred, target)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            dice += metric.binary.dc(pred, target)
            count = count + 1
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        asd = asd / count
        dice = 100 * dice / count
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/ASD', asd, epoch)
        self.writer.add_scalar('val/dsc', dice, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{},fwIoU: {}, asd: {}, dsc: {}".format(Acc, Acc_class, mIoU,
                                                                                 FWIoU, asd, dice))
        print('Loss: %.3f' % test_loss)

        # save the best model,new pred is optional,like dice,Miou...
        new_pred = dice
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


if __name__ == "__main__":
    main()
