import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model1')
parser.add_argument('dataset')
parser.add_argument('ratio', type=int)
parser.add_argument('output')
parser.add_argument('--rep', type=int, default=0)
parser.add_argument('--datapath', default='/data')
parser.add_argument('--imgsize', default=768, type=int)
parser.add_argument('--epochs', default=20, type=int)  # 100
parser.add_argument('--selection-method', default='random_sample', choices=['random_sample', 'completely_random', 'choose_best'])
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
from time import time
import data, utils
device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################## DATA ##################################

ds = getattr(data, args.dataset)
# for simplicity patch determined from the ratio
patchsize = args.imgsize // args.ratio // args.ratio

l = []
if ds.can_rotate:
    l += [A.Rotate(180)]
l += [
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(0.1),
    A.Resize(int(args.imgsize*1.1), int(args.imgsize*1.1)),
    A.RandomCrop(args.imgsize, args.imgsize),
    A.Normalize(0, 1),
    ToTensorV2()
]
transform = A.Compose(l)

tr = ds(args.datapath, 'train', args.rep, transform)
tr = torch.utils.data.DataLoader(tr, 8, True, num_workers=4, pin_memory=True)

################################## MODEL ##################################

model1 = torch.load(args.model1, map_location=device)
#model2 = unet.UNet(ds.colors, True).to(device)
# train model2 as transfer-learning from model1
model2 = torch.load(args.model1)
#model2.final = torch.nn.Conv2d(32*2, 1, 3, 1, 1)
model2.to(device)
opt = torch.optim.Adam(model2.parameters(), 1e-4)
loss_fn = lambda ypred, y: torchvision.ops.sigmoid_focal_loss(ypred, y, reduction='mean')
selection_method = getattr(utils, args.selection_method)

################################## LOOP ##################################

model1.eval()
model2.train()
full_time = 0
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    avg_acc0 = 0
    avg_acc1 = 0
    for img, seg in tr:
        img = img.to(device)
        seg = seg.to(device)[:, None]

        size1 = (args.imgsize//args.ratio, args.imgsize//args.ratio)
        img1 = torchvision.transforms.functional.resize(img, size1)
        with torch.no_grad():
            pred1, pre_pred1 = model1(img1)
        npatches = 1
        ix, img2, seg2 = utils.sample_patches(img, seg, pred1, args.ratio, patchsize, npatches, selection_method)

        mult = patchsize
        pred1_hi = torch.stack([torchvision.transforms.functional.resize(pre_pred1[i, :, mult*y:mult*(y+1), mult*x:mult*(x+1)], size1, torchvision.transforms.InterpolationMode.NEAREST) for i, (yy, xx) in enumerate(ix) for y, x in zip(yy, xx)])

        #pred2 = model2(img2, embed.repeat(npatches, 1, 1, 1))['out']
        #pred2 = model2(img2, embed)['out']
        #pred2 = model2(torch.cat((img2, seg2), 1))['out']
        #pred2 = model2(img2)['out']
        pred2, _ = model2(img2, pred1_hi)

        loss = loss_fn(pred2, seg2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
        avg_acc0 += float(torch.sum((pred2 < 0) & (seg2 == 0)) / torch.sum(seg2 == 0)) / len(tr)
        avg_acc1 += float(torch.sum((pred2 > 0) & (seg2 == 1)) / torch.sum(seg2 == 1)) / len(tr)
    toc = time()
    full_time += toc-tic
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Loss: {avg_loss} - Acc0: {avg_acc0} - Acc1: {avg_acc1}')

with open('time-train2.csv', 'a') as f:
    print(args.output, full_time, file=f, sep=',')
torch.save(model2.cpu(), args.output)
