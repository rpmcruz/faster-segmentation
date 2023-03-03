import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('rep', type=int)
parser.add_argument('ratio', type=int)
parser.add_argument('model')
parser.add_argument('--datapath', default='/data')
parser.add_argument('--imgsize', default=768, type=int)
parser.add_argument('--epochs', default=50, type=int)
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from time import time
import data, metrics
device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################## DATA ##################################

ds = getattr(data, args.dataset)
img_size = args.imgsize//args.ratio

l = [
    A.Resize(args.imgsize, args.imgsize),
    A.Normalize(0, 1),
    ToTensorV2()
]
transform = A.Compose(l)

ts = ds(args.datapath, 'test', args.rep, transform)
ts = torch.utils.data.DataLoader(ts, 1, num_workers=4, pin_memory=True)

################################## MODEL ##################################

model = torch.load(args.model, map_location=device)
opt = torch.optim.Adam(model.parameters(), 1e-4)
loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

################################## LOOP ##################################

model.eval()
avg_dice = 0
full_time = 0
for img, seg in ts:
    tic = time()
    img = img.to(device)
    seg = seg.to(device)[:, None]
    with torch.no_grad():
        pred, _ = model(img)
    pred = torch.sigmoid(pred) >= 0.5
    avg_dice += float(metrics.dice_score(pred, seg)) / len(ts)
    toc = time()
    full_time += toc-tic

print(args.model, avg_dice, full_time, sep=',')
