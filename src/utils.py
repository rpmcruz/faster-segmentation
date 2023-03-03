import torch
import numpy as np

def to_uncertainty(pred):
    pred = torch.sigmoid(pred)
    return 1 - torch.abs(0.5 - pred)*2

def completely_random(p, n):
    return np.random.choice(p.size, n, False)

def random_sample(p, n):
    return np.random.choice(p.size, n, False, p)

def choose_best(p, n):
    return np.argsort(-p)[:n]

def threshold(p, th):
    # for this one, you should use normalized_probs=False
    return np.where(p > th)[0]

def sample_indices(pred1, patchsize, npatches, method, normalized_probs):
    uncertain1 = to_uncertainty(pred1)
    uncertain1 = torch.nn.functional.avg_pool2d(uncertain1, patchsize, patchsize, 1, count_include_pad=False)
    probs = torch.flatten(uncertain1, 1)
    if normalized_probs:
        # normalize probabilities across regions
        probs = probs / torch.sum(probs, 1, True)
        # this may not sum exactly to 1 (clamp necessary due to floating errors)
        probs[:, -1] = torch.clamp(1-probs[:, :-1].sum(1), 0, 1)
        probs[torch.isnan(probs)] = 1/probs.shape[1]
    else:
        probs[torch.isnan(probs)] = 0
    probs = probs.cpu().numpy()
    ix = [method(p, npatches) for p in probs]
    ix = [(i // uncertain1.shape[3], i % uncertain1.shape[3]) for i in ix]
    return ix

def sample_patches(img, seg, pred1, ratio, patchsize, npatches, method, normalized_probs):
    if method != threshold and npatches == 0: return [], None, None
    ix = sample_indices(pred1, patchsize, npatches, method, normalized_probs)
    # the following condition makes this function only appropriate for 1 image
    if len(ix[0][0]) == 0: return [], None, None
    mult = ratio * patchsize
    img2 = torch.stack([img[i, :, mult*y:mult*(y+1), mult*x:mult*(x+1)] for i, (yy, xx) in enumerate(ix) for y, x in zip(yy, xx)])
    seg2 = torch.stack([seg[i, :, mult*y:mult*(y+1), mult*x:mult*(x+1)] for i, (yy, xx) in enumerate(ix) for y, x in zip(yy, xx)])
    return ix, img2, seg2
