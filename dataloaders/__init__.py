from dataloaders.datasets import medical
from torch.utils.data import DataLoader


def data_loader(args, **kwargs):
    if args.dataset == 'medical':
        train_set = medical.MedicalSegmentDataset(args, split='FS')
        val_set = medical.MedicalSegmentDataset(args, split='AH')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, num_class

    elif args.dataset == 'others':
        # other custom dataset
        pass

    else:
        raise NotImplementedError
