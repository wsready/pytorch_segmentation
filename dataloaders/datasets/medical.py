# @Time    : 2023/8/29 23:11
# @Author  : wang song
# @File    : medical.py
# @Description :
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders.transforms import Normalize, ToTensor, FixedResize


class MedicalSegmentDataset(Dataset):

    NUM_CLASSES = 2

    def __init__(self, args, split=None, transforms=None):
        self.base_dir = args.base_dir
        self.split = split
        assert os.path.exists(self.base_dir), f"path '{self.base_dir}' does not exist."
        if self.split == 'FS':
            self.image_root = os.path.join(self.base_dir, self.split, "images")
            self.mask_root = os.path.join(self.base_dir, self.split, "masks")
        else:
            self.image_root = os.path.join(self.base_dir, self.split, "images")
            self.mask_root = os.path.join(self.base_dir, self.split, "masks")
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root)]
        mask_names = [p for p in os.listdir(self.mask_root)]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check image and mask
        re_mask_names = []
        for mask_name in image_names:
            assert mask_name in mask_names, f"{mask_name} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        # image = Image.open(self.images_path[idx]).convert('RGB')
        # mask = Image.open(self.masks_path[idx])
        # mask = mask.convert("L")
        # mask = np.array(mask)
        # mask[mask < 128] = 0
        # mask[mask >= 128] = 1
        #
        # mask = Image.fromarray(mask)
        # if self.split == 'AH' or self.split == 'val':
        #     sample = {'image': image, 'label': mask}
        #     if self.transforms:
        #         sample = self.transform_val(sample)
        #         # sample['filename'] = os.path.basename(self.images_path[idx])
        #     return sample
        # else:
        #     sample = {'image': image, 'label': mask}
        #     if self.transforms:
        #         sample = self.transform_val(sample)
        #         # sample['filename'] = os.path.basename(self.images_path[idx])
        #     return sample
        img = Image.open(self.images_path[idx]).convert('RGB')
        mask = Image.open(self.masks_path[idx])
        mask = mask.convert("L")
        mask = np.array(mask)
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        # print(mask)
        mask = Image.fromarray(mask)

        sample = {'image': img, 'label': mask}

        # if self.transforms is not None:
        #     img, mask = self.transforms(img, mask)
        sample = self.transform_tr(sample)
        # return img, mask
        sample['filename'] = os.path.basename(self.images_path[idx])
        return sample

    def __len__(self):
        return len(self.images_path)

    # transform for train set
    def transform_tr(self, sample):
        Medical_MEAN = (0.485, 0.456, 0.406)
        Medical_STD = (0.229, 0.224, 0.225)

        # transform = transforms.Compose(
        #     [FixedResize(512),RandomHorizontalFlip(), RandomGaussianBlur(), Normalize(mean=Medical_MEAN, std=Medical_STD), ToTensor()]
        # )
        transform = transforms.Compose(
            [FixedResize(512), Normalize(mean=Medical_MEAN, std=Medical_STD), ToTensor()]
        )

        return transform(sample)

    # transform for val set
    def transform_val(self, sample):
        Medical_MEAN = (0.485, 0.456, 0.406)
        Medical_STD = (0.229, 0.224, 0.225)

        # transform = transforms.Compose(
        #     [FixedResize(512),RandomHorizontalFlip(), RandomGaussianBlur(), Normalize(mean=Medical_MEAN, std=Medical_STD), ToTensor()]
        # )
        transform = transforms.Compose(
            [FixedResize(512), Normalize(mean=Medical_MEAN, std=Medical_STD), ToTensor()]
        )

        return transform(sample)
