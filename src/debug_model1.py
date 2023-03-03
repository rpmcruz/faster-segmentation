import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('ratio', type=int)
parser.add_argument('npatches', type=int)
parser.add_argument('method', choices=['static', 'random'])
parser.add_argument('model1')
parser.add_argument('--datapath', default='/data')
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import data, utils
device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################## DATA ##################################

ds = getattr(data, args.dataset)
# for simplicity patch determined from the ratio
patchsize = ds.hi_size // args.ratio // args.ratio

l = [
    A.Resize(ds.hi_size, ds.hi_size),
    A.Normalize(0, 1),
    ToTensorV2()
]
transform = A.Compose(l)

ts = ds(args.datapath, 'test', transform)
ts = torch.utils.data.DataLoader(ts, 1)

################################## MODEL ##################################

model1 = torch.load(args.model1).to(device)

################################## LOOP ##################################

model1.eval()
for img, seg in ts:
    img = img.to(device)
    seg = seg.to(device)[:, None]

    size1 = (ds.hi_size//args.ratio, ds.hi_size//args.ratio)
    img1 = torchvision.transforms.functional.resize(img, size1)
    with torch.no_grad():
        pred1 = model1(img1)['out']

    seg_pred = torch.sigmoid(pred1)
    method = getattr(utils, f'{args.method}_sample')
    ix = utils.sample_indices(pred1, patchsize, args.npatches, method)
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.imshow(img[0].permute(1, 2, 0).cpu())
    plt.title('Image')
    plt.subplot(1, 3, 2)
    plt.imshow(seg[0, 0].cpu(), cmap='gray_r')
    plt.title('Ground-truth')
    plt.subplot(1, 3, 3)
    plt.imshow(seg_pred[0, 0].cpu(), cmap='gray_r')
    plt.title('Prediction')
    mult = patchsize

    conf = torch.nn.functional.avg_pool2d(utils.to_conf(pred1), patchsize, patchsize, 1, count_include_pad=False)
    for i in range(conf.shape[-1]):
        for j in range(conf.shape[-2]):
            plt.text((i+0.5)*mult, (j+0.5)*mult, str(int(conf[:, :, j, i]*100)), c='blue', size=8, ha='center', va='center')

    yy, xx = ix[0]
    for y, x in zip(yy, xx):
        plt.gca().add_patch(Rectangle((x*mult, y*mult), mult, mult, linewidth=3, edgecolor='red', facecolor='none'))
    plt.suptitle(f'Method = {args.method}')
    plt.show()
