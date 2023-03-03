import torch
import torchvision

# The first two models come from pytorch. The UNet is implemented by us,
# but also uses Resnet50 as the backbone.

class Deeplabv3_Resnet50(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
        if in_channels == 1:
            self.model.backbone.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.model.classifier[4] = torch.nn.Identity()
        self.final = torch.nn.Conv2d(256, 1, 3, 1, 1)

    def forward(self, x, out2=None):
        x = pre_x = self.model(x)['out']
        if out2 is not None:
            x = x + out2
        x = self.final(x)
        return x, pre_x

class FCN_Resnet50(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = torchvision.models.segmentation.fcn_resnet50(weights='DEFAULT')
        if in_channels == 1:
            self.model.backbone.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.model.classifier[4] = torch.nn.Identity()
        self.final = torch.nn.Conv2d(512, 1, 3, 1, 1)

    def forward(self, x, out2=None):
        x = pre_x = self.model(x)['out']
        if out2 is not None:
            x = x + out2
        x = self.final(x)
        return x, pre_x

class UNet_Resnet50(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        encoder = torchvision.models.resnet50(weights='DEFAULT')
        self.encoders = torch.nn.ModuleList([
            # omitting maxpool layer to avoid double downsample
            torch.nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu),
            # introducing maxpool layer here to make one downsample
            torch.nn.Sequential(encoder.layer1, encoder.maxpool),
            encoder.layer2,
            encoder.layer3,
            encoder.layer4
        ])
        self.decoders = torch.nn.ModuleList([
            torch.nn.Conv2d(2048*2, 1024, 3, padding=1),
            torch.nn.Conv2d(1024*2, 512, 3, padding=1),
            torch.nn.Conv2d(512*2, 256, 3, padding=1),
            torch.nn.Conv2d(256*2, 64, 3, padding=1),
            torch.nn.Conv2d(64*2, 32, 3, padding=1)
        ])
        self.final = torch.nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, out2=None):
        sizes = []
        skip = []
        for e in self.encoders:
            sizes.append(x.shape[2:])
            x = e(x)
            skip.append(x)
        for s, sz, d in zip(skip[::-1], sizes[::-1], self.decoders):
            x = torch.cat((s, x), 1)
            x = d(x)
            x = torch.nn.functional.interpolate(x, sz)
        pre_x = x
        if out2 is not None:
            x = x + out2
        x = self.final(x)
        return x, pre_x

class SegNet_Resnet50(torch.nn.Module):
    # Same as the UNet, but without the skip connections
    def __init__(self, in_channels):
        super().__init__()
        encoder = torchvision.models.resnet50(weights='DEFAULT')
        # ResNet50 has blocks with double downsample, eliminate them
        #encoder.layer2[0].downsample = torch.nn.Identity()
        self.encoders = torch.nn.ModuleList([
            # omitting maxpool layer to avoid double downsample
            torch.nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu),
            # introducing maxpool layer here to make one downsample
            torch.nn.Sequential(encoder.layer1, encoder.maxpool),
            encoder.layer2,
            encoder.layer3,
            encoder.layer4
        ])
        self.decoders = torch.nn.ModuleList([
            torch.nn.Conv2d(2048, 1024, 3, padding=1),
            torch.nn.Conv2d(1024, 512, 3, padding=1),
            torch.nn.Conv2d(512, 256, 3, padding=1),
            torch.nn.Conv2d(256, 64, 3, padding=1),
            torch.nn.Conv2d(64, 32, 3, padding=1)
        ])
        self.final = torch.nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, out2=None):
        sizes = []
        for e in self.encoders:
            sizes.append(x.shape[2:])
            x = e(x)
        for sz, d in zip(sizes[::-1], self.decoders):
            x = d(x)
            x = torch.nn.functional.interpolate(x, sz)
        pre_x = x
        if out2 is not None:
            x = x + out2
        x = self.final(x)
        return x, pre_x

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--imgsize', default=768, type=int)
    args = parser.parse_args()
    from torchsummary import summary
    from thop import profile
    if args.model == 'backbone':
        model = torchvision.models.resnet50(weights='DEFAULT')
        model = torch.nn.Sequential(*list(model.children())[:-2])
    else:
        model = globals()[args.model](3)
    for ratio in [1, 2, 4, 8, 16]:
        imgsize = args.imgsize//ratio
        flops, params = profile(model, [torch.zeros(1, 3, imgsize, imgsize)], verbose=False)
        print('ratio:', ratio, '#params:', params/1e6, 'Mflops:', round(flops/1e6))
    for ratio in [1, 2, 4, 8, 16]:
        imgsize = args.imgsize//ratio
        summary(model, (3, imgsize, imgsize))
