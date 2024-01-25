import os
import shutil
import torch
from collections import OrderedDict
import glob


class Saver(object):

    def __init__(self, args):

        self.args = args
        self.directory = os.path.join('results', args.dataset, args.checkname)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.directory, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.directory, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            previous_miou = [0.0]
            path = os.path.join(self.directory, 'best_pred.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    miou = float(f.readline())
                    previous_miou.append(miou)
            max_miou = max(previous_miou)
            if best_pred > max_miou:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))

    def save_experiment_config(self):
        logfile = os.path.join(self.directory, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['outstride'] = self.args.outstride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
