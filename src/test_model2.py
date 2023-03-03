import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('rep', type=int)
parser.add_argument('ratio', type=int)
parser.add_argument('npatches', type=int)
parser.add_argument('model1')
parser.add_argument('model2')
parser.add_argument('--threshold', type=float)
parser.add_argument('--datapath', default='/data')
parser.add_argument('--imgsize', default=768, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--savefigs', action='store_true')
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
from time import time
import data, utils, metrics
device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################## DATA ##################################

ds = getattr(data, args.dataset)
# for simplicity patch determined from the ratio
patchsize = args.imgsize // args.ratio // args.ratio

l = [
    A.Resize(args.imgsize, args.imgsize),
    A.Normalize(0, 1),
    ToTensorV2()
]
transform = A.Compose(l)

ts = ds(args.datapath, 'test', args.rep, transform)
ts = torch.utils.data.DataLoader(ts, 1, num_workers=4, pin_memory=True)

################################## MODEL ##################################

model1 = torch.load(args.model1, map_location=device)
model2 = torch.load(args.model2, map_location=device)

################################## LOOP ##################################

model1.eval()
model2.eval()
avg_dice = 0
full_time = 0
for loopi, (img, seg) in enumerate(ts):
    tic = time()
    img = img.to(device)
    seg = seg.to(device)[:, None]

    size1 = (args.imgsize//args.ratio, args.imgsize//args.ratio)
    img1 = torchvision.transforms.functional.resize(img, size1)
    with torch.no_grad():
        pred1, pre_pred1 = model1(img1)

    if args.threshold is not None:
        ix, img2, seg2 = utils.sample_patches(img, seg, pred1, args.ratio, patchsize, args.threshold, utils.threshold, False)
    else:
        ix, img2, seg2 = utils.sample_patches(img, seg, pred1, args.ratio, patchsize, args.npatches, utils.choose_best, True)

    if len(ix) == 0:
        # special case: only evaluate model1
        pred1 = torch.sigmoid(pred1)
        pred = torchvision.transforms.functional.resize(pred1, img.shape[2:])
        avg_dice += float(metrics.dice_score(pred >= 0.5, seg)) / len(ts)
        toc = time()
        full_time += toc-tic
        continue

    mult = patchsize
    '''
    pred1_hi = []
    print('pred1:', pred1.shape)
    import matplotlib.pyplot as plt
    tempi = 0
    for i, (yy, xx) in enumerate(ix):
        for y, x in zip(yy, xx):
            temp = torchvision.transforms.functional.resize(pred1[i, :, mult*y:mult*(y+1), mult*x:mult*(x+1)], size1, torchvision.transforms.InterpolationMode.NEAREST)
            print('each', i, temp.shape, 'y:', mult*y, mult*(y+1), 'x:', mult*x, mult*(x+1))
            pred1_hi.append(temp)
            plt.subplot(2, 5, tempi+1)
            plt.imshow(temp[0].cpu(), vmin=0, vmax=1)
            tempi += 1
    pred1_hi = torch.stack(pred1_hi)
    print('pred1_hi:', pred1_hi.shape)
    plt.show()
    '''
    pred1_hi = torch.stack([torchvision.transforms.functional.resize(pre_pred1[i, :, mult*y:mult*(y+1), mult*x:mult*(x+1)], size1, torchvision.transforms.InterpolationMode.NEAREST) for i, (yy, xx) in enumerate(ix) for y, x in zip(yy, xx)])

    with torch.no_grad():
        #pred2 = model2(img2, embed.repeat(args.npatches, 1, 1, 1))['out']
        #pred2 = model2(img2)['out']
        pred2, _ = model2(img2, pred1_hi)
        #pred2 = pred1_hi

    pred1 = torch.sigmoid(pred1)
    pred2 = torch.sigmoid(pred2)

    # must resize pred1 to the same size as img => pred
    pred = torchvision.transforms.functional.resize(pred1, img.shape[2:], torchvision.transforms.InterpolationMode.NEAREST)
    #pred = pred >= 0.5  # just to ensure no error on interpolation

    pred_before = pred.clone()
    mult = args.ratio * patchsize
    for i, (yy, xx) in enumerate(ix):
        for j, (y, x) in enumerate(zip(yy, xx)):
            pred[i, :, mult*y:mult*(y+1), mult*x:mult*(x+1)] = pred2[i*args.npatches+j]

    '''
    import matplotlib.pyplot as plt
    plt.subplot(2, 2, 1)
    plt.imshow(img[0].permute(1, 2, 0).cpu())
    plt.subplot(2, 2, 2)
    plt.imshow(seg[0, 0].cpu(), vmin=0, vmax=1)
    plt.subplot(2, 2, 3)
    plt.imshow(pred_before[0, 0].cpu(), vmin=0, vmax=1)
    plt.subplot(2, 2, 4)
    plt.imshow(pred[0, 0].cpu(), vmin=0, vmax=1)
    plt.show()
    '''

    avg_dice += float(metrics.dice_score(pred >= 0.5, seg)) / len(ts)

    if args.savefigs and loopi < 25:
        from skimage import draw
        from skimage.io import imsave
        import os
        fname = f'output-{os.path.basename(args.model1)}-{os.path.basename(args.model2)}-{args.npatches}-{loopi:02d}'
        imsave(f'{fname}-img.png', (img[0].permute(1, 2, 0).cpu()*255).byte(), check_contrast=False)
        imsave(f'{fname}-gt.png', 255-(seg[0, 0].cpu()*255).byte(), check_contrast=False)
        imsave(f'{fname}-model1-proba.png', 255-(pred1[0, 0].cpu()*255).byte(), check_contrast=False)
        imsave(f'{fname}-model1.png', 255-((pred1[0, 0].cpu() >= 0.5)*255).byte(), check_contrast=False)
        model2_out = 255-((pred[0, 0].cpu()[..., None].repeat(1, 1, 3) >= 0.5)*255).byte()
        for yy, xx in ix:
            for y, x in zip(yy, xx):
                rr, cc = draw.rectangle_perimeter((x*mult+2, y*mult+2), extent=(mult-5-2, mult-5-2), shape=model2_out.shape)
                for xshift in range(5):
                    for yshift in range(5):
                        _cc = (cc+yshift).clip(0, model2_out.shape[0]-1)
                        _rr = (rr+xshift).clip(0, model2_out.shape[1]-1)
                        color = torch.tensor((0, 0, 255), dtype=torch.uint8)
                        model2_out[_cc, _rr] = color
        imsave(f'{fname}-model2.png', model2_out, check_contrast=False)
        '''
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import os
        plt.clf()
        plt.rcParams['figure.figsize'] = (20, 20)
        plt.rcParams['backend'] = 'Agg'
        plt.subplot(2, 2, 1)
        plt.imshow(img[0].permute(1, 2, 0).cpu())
        plt.title('Image')
        plt.subplot(2, 2, 2)
        plt.imshow(seg[0, 0].cpu(), cmap='gray_r', vmin=0, vmax=1)
        plt.title('Ground-truth')
        plt.subplot(2, 2, 3)
        plt.imshow(pred1[0, 0].cpu() >= 0.5, cmap='gray_r', vmin=0, vmax=1)
        plt.title('Model 1')
        plt.subplot(2, 2, 4)
        plt.imshow(pred[0, 0].cpu() >= 0.5, cmap='gray_r', vmin=0, vmax=1)
        for yy, xx in ix:
            for y, x in zip(yy, xx):
                plt.gca().add_patch(Rectangle((x*mult, y*mult), mult, mult, linewidth=3, edgecolor='red', facecolor='none'))
        plt.title('Model 1 + 2')
        pred1_hi_ = torchvision.transforms.functional.resize(pred1, (args.imgsize, args.imgsize), torchvision.transforms.InterpolationMode.NEAREST)
        plt.suptitle(f'Dice = {metrics.dice_score(pred1_hi_ >= 0.5, seg):.3f} vs {metrics.dice_score(pred >= 0.5, seg):.3f}')
        plt.savefig(f'output-{os.path.basename(args.model1)}-{os.path.basename(args.model2)}-{args.npatches}-{loopi:02d}.png')
        '''
    toc = time()
    full_time += toc-tic
v = args.npatches if args.threshold is None else args.threshold
print(args.model1, args.model2, v, avg_dice, full_time, sep=',')
