import torch.nn as nn
import torch
import torchvision

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


def define_F_54(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

def define_F_44(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    feature_layer = 25
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

def define_F_34(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    feature_layer = 16
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

def define_F_22(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    feature_layer = 7
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

def define_F_12(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    feature_layer = 2
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF


class CondNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, opt):
        super(CondNet, self).__init__()
        self.netF_44 = define_F_44(opt, use_bn=False)
        self.netF_34 = define_F_34(opt, use_bn=False)
        self.netF_22 = define_F_22(opt, use_bn=False)
        self.netF_12 = define_F_12(opt, use_bn=False)
        self.tconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.tconv3 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.tconv5 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.tconv7 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.tconv8 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.output = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        residual_1 = self.netF_12(x.repeat((1, 3, 1, 1)))
        residual_2 = self.netF_22(x.repeat((1, 3, 1, 1)))
        residual_3 = self.netF_34(x.repeat((1, 3, 1, 1)))
        residual_4 = self.netF_44(x.repeat((1, 3, 1, 1)))
        out = self.relu(self.tconv1(residual_4))
        out = self.relu(self.tconv2(out))
        out = torch.cat([residual_3, out], 1)
        out = self.relu(self.tconv3(out))
        out = self.relu(self.tconv4(out))
        out = torch.cat([residual_2, out], 1)
        out = self.relu(self.tconv5(out))
        out = self.relu(self.tconv6(out))
        out = torch.cat([residual_1, out], 1)
        out = self.relu(self.tconv7(out))
        out = self.relu(self.tconv8(out))
        out = self.output(out)
        out = self.relu(out)
        return out


