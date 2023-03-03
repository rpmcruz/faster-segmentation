import argparse
parser = argparse.ArgumentParser()
parser.add_argument('architecture', choices=['Deeplabv3_Resnet50', 'FCN_Resnet50', 'UNet_Resnet50', 'SegNet_Resnet50'])
parser.add_argument('dataset')
parser.add_argument('ratio', type=int)
parser.add_argument('output')
parser.add_argument('--rep', type=int, default=0)
parser.add_argument('--datapath', default='/data')
parser.add_argument('--imgsize', default=768, type=int)
parser.add_argument('--epochs', default=20, type=int)  # 100
parser.add_argument('--batchsize', default=8, type=int)  # 100
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
from time import time
import data, models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################## DATA ##################################

ds = getattr(data, args.dataset)
img_size = args.imgsize//args.ratio

l = []
if ds.can_rotate:
    l += [A.Rotate(180)]
l += [
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(0.1),
]
l += [
    A.Resize(int(img_size*1.1), int(img_size*1.1)),
    A.RandomCrop(img_size, img_size),
]
l += [
    A.Normalize(0, 1),
    ToTensorV2()
]
transform = A.Compose(l)

tr = ds(args.datapath, 'train', args.rep, transform)
tr = torch.utils.data.DataLoader(tr, args.batchsize, True, num_workers=4, pin_memory=True)

################################## MODEL ##################################

model = getattr(models, args.architecture)(ds.colors).to(device)
opt = torch.optim.Adam(model.parameters(), 1e-4)
loss_fn = lambda ypred, y: torchvision.ops.sigmoid_focal_loss(ypred, y, reduction='mean')
# + (1-metrics.dice_score(torch.sigmoid(ypred), y))

################################## LOOP ##################################

model.train()
full_time = 0
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    avg_acc0 = 0
    avg_acc1 = 0
    for img, seg in tr:
        img = img.to(device)
        seg = seg.to(device)[:, None]
        pred, _ = model(img)

        loss = loss_fn(pred, seg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(tr)
        avg_acc0 += float(torch.sum((pred < 0) & (seg == 0)) / torch.sum(seg == 0)) / len(tr)
        avg_acc1 += float(torch.sum((pred > 0) & (seg == 1)) / torch.sum(seg == 1)) / len(tr)
    toc = time()
    full_time += toc-tic
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Loss: {avg_loss} - Acc0: {avg_acc0} - Acc1: {avg_acc1}')

with open('time-train1.csv', 'a') as f:
    print(args.output, full_time, file=f, sep=',')
torch.save(model.cpu(), args.output)
