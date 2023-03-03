from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import os

def train_test_split(data, fold, rep):
    rand = np.random.RandomState(123+rep)
    ix = rand.choice(len(data), len(data), False)
    if fold == 'train':
        ix = ix[:int(0.70*len(data))]
    else:
        ix = ix[int(0.70*len(data)):]
    return [data[i] for i in ix]

class BDD(Dataset):
    # https://www.bdd100k.com/

    nclasses = 2
    colors = 3
    can_rotate = False

    def __init__(self, root, fold, rep=None, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        fold = fold if fold == 'train' else 'val'
        self.root_img = os.path.join(root, 'bdd100k', 'images', '10k', fold)
        self.root_seg = os.path.join(root, 'bdd100k', 'labels', 'sem_seg', 'masks', fold)
        self.files = sorted(os.listdir(self.root_seg))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        img = imread(os.path.join(self.root_img, f'{fname[:-3]}jpg'))
        seg = imread(os.path.join(self.root_seg, fname))
        seg[seg == 255] = 0
        seg = np.logical_and(seg >= 13, seg <= 16).astype(np.float32)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img, seg = d['image'], d['mask']
        return img, seg

class BOWL2018(Dataset):
    # https://www.kaggle.com/c/data-science-bowl-2018

    nclasses = 2
    colors = 3
    can_rotate = True

    def __init__(self, root, fold, rep, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        self.fold = fold
        self.root = os.path.join(root, 'data-science-bowl-2018', 'stage1', 'train')
        self.dirs = os.listdir(self.root)
        self.transform = transform
        self.dirs = train_test_split(self.dirs, fold, rep)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        dir = self.dirs[i]
        img = imread(os.path.join(self.root, dir, 'images', dir + '.png'))
        if img.shape[-1] == 4:
            img = img[..., :3]
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, -1)
        masks = []
        for mask in os.listdir(os.path.join(self.root, dir, 'masks')):
            mask = os.path.join(self.root, dir, 'masks', mask)
            mask = imread(mask, True) >= 128
            masks.append(mask)
        seg = np.clip(np.sum(masks, 0), 0, 1).astype(np.float32)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img, seg = d['image'], d['mask']
        return img, seg

class KITTI(Dataset):
    # https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

    nclasses = 2
    colors = 3
    can_rotate = False

    def __init__(self, root, fold, rep, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        self.fold = fold
        self.root = os.path.join(root, 'kitti', 'semantics', 'training')
        self.files = sorted(os.listdir(os.path.join(self.root, 'semantic')))
        self.transform = transform
        self.files = train_test_split(self.files, fold, rep)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        img = imread(os.path.join(self.root, 'image_2', fname))
        seg = imread(os.path.join(self.root, 'semantic', fname))
        # 26-31 are cars, trucks and etc
        # 32=motorcycle, 33=bicycle, 25=cyclist, 24=pedestrian
        seg = np.logical_and(seg >= 26, seg <= 31).astype(np.float32)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img = d['image']
            seg = d['mask']
        return img, seg

class PH2(Dataset):
    # https://www.fc.up.pt/addi/ph2%20database.html

    nclasses = 2
    colors = 3
    can_rotate = True

    def __init__(self, root, fold, rep, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        self.root = os.path.join(root, 'PH2Dataset', 'PH2 Dataset images')
        self.files = sorted(os.listdir(self.root))
        self.transform = transform
        self.files = train_test_split(self.files, fold, rep)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        img = imread(os.path.join(self.root, fname, f'{fname}_Dermoscopic_Image', f'{fname}.bmp'))
        seg = (imread(os.path.join(self.root, fname, f'{fname}_lesion', f'{fname}_lesion.bmp'), True) >= 128).astype(np.float32)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img, seg = d['image'], d['mask']
        return img, seg

class RETINA(Dataset):
    # concatenation of the following datasets:
    # https://blogs.kingston.ac.uk/retinal/chasedb1/
    # https://cecas.clemson.edu/~ahoover/stare/
    # https://drive.grand-challenge.org/

    nclasses = 2
    colors = 3
    can_rotate = True

    def __init__(self, root, fold, rep, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        self.root = os.path.join(root, 'RETINA')
        self.files = [f[:-9] for f in sorted(os.listdir(self.root)) if f.endswith('image.png')]
        self.transform = transform
        self.files = train_test_split(self.files, fold, rep)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        img = imread(os.path.join(self.root, f'{fname}image.png'))
        seg = (imread(os.path.join(self.root, f'{fname}mask.png'), True) >= 128).astype(np.float32)
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img, seg = d['image'], d['mask']
        return img, seg

class SARTORIUS(Dataset):
    # https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation
    # kaggle competitions download -c sartorius-cell-instance-segmentation

    nclasses = 2
    colors = 3 #1
    can_rotate = True

    def __init__(self, root, fold, rep, transform=None):
        assert fold in ['train', 'test'], f'fold {fold} must be train or test'
        self.fold = fold
        self.root = os.path.join(root, 'sartorius-cell-instance-segmentation')
        self.transform = transform

        f = np.loadtxt(os.path.join(self.root, 'train.csv'), str, delimiter=',', skiprows=1, usecols=[0, 1, 2, 3])
        self.segs = {}
        for id, ann, w, h in f:
            if id not in self.segs:
                self.segs[id] = np.zeros((int(h), int(w)), np.float32)
            SARTORIUS.annotation(ann, self.segs[id])
        self.ids = list(self.segs)
        self.ids = train_test_split(self.ids, fold, rep)

    @staticmethod
    def annotation(s, seg):
        starts = [int(x)-1 for x in s.split()[::2]]
        lengths = [int(y) for y in s.split()[1::2]]
        for s, l in zip(starts, lengths):
            seg.flat[s:s+l] = 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        img = imread(os.path.join(self.root, 'train', id + '.png'))
        if len(img.shape) == 2:
            img = gray2rgb(img)
        elif img.shape[2] == 4:
            img = img[..., :3]
        seg = self.segs[id]
        if self.transform:
            d = self.transform(image=img, mask=seg)
            img, seg = d['image'], d['mask']
        return img, seg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--datapath', default='/data')
    args = parser.parse_args()

    # sizes
    ds = globals()[args.dataset]
    print(args.dataset, 'train', len(ds(args.datapath, 'train', 0)))
    print(args.dataset, 'test', len(ds(args.datapath, 'test', 0)))

    import matplotlib.pyplot as plt
    ds = globals()[args.dataset]
    ds = ds(args.datapath, 'train', 0)
    imbalance = 0
    for img, seg in ds:
        imbalance += seg.mean() / len(ds)
    print(args.dataset, imbalance)
    '''
    img, seg = ds[0]
    print('img:', img.shape, img.dtype, img.min(), img.max())
    print('seg:', seg.shape, seg.dtype, seg.min(), seg.max())
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(seg)
    plt.show()
    '''
