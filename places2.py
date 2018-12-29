import random
import torch
from PIL import Image
from glob import glob


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            #self.paths = glob('{:s}/data_large/**/*.jpg'.format(img_root), recursive=True) ## origin
            print("img_root",img_root)
            self.paths = glob('./srv/datasets/Places2/data_all/**/*.jpg'.format(img_root), recursive=True)
        else:
            self.paths = glob('{:s}/{:s}_large/*'.format(img_root, split)) ## origin
            self.paths = glob('./srv/datasets/Places2/val_ijk/**/*.jpg')
            #self.paths = glob('{./data/*.jpg')
            #print("paths is ",self.paths)

        #self.mask_paths = glob('./srv/datasets/Places2/*.jpg'.format(mask_root))
        print("mask_root%%%%%%",mask_root)
        self.mask_paths = glob('{:s}/*.jpg'.format(mask_root)) ## origin
        self.N_mask = len(self.mask_paths)
        print("number of mask",self.N_mask)

    def __getitem__(self, index):
        #print("Now path is ",self.paths)
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
