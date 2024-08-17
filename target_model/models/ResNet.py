import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# TODO: 可以再优化的，client端模型不保留server 参数
__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
]
# split layer [2,3,5,7,9,11,12,13]

resnet_model_cfg = {
    'resnet18': [
        ("conv1", 2),
        ("pooling", 3),
        ("layer11", 5),
        ("layer21", 7),
        ("layer31", 9),
        ("layer41", 11),
        ("avgpool", 12),
        ("fc", 13),
    ]
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        activation='relu'
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.relu = nn.GELU()
        elif activation == 'tanh':
            self.relu = nn.Tanh()
        else:
            raise AssertionError()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def forward_for_jacobian(self, x):
        identity = x

        out = self.conv1(x)
        # BN must be handled for torchfunc jvp
        out = out.unsqueeze(0)
        out = self.bn1(out)
        out = out.squeeze()
        out = self.relu(out)

        out = self.conv2(out)
        out = out.unsqueeze(0)
        out = self.bn2(out)
        out = out.squeeze()

        if self.downsample is not None:
            for layer in self.downsample:
                if isinstance(layer, nn.BatchNorm2d):
                    identity = identity.unsqueeze(0)
                    identity = layer(identity)
                    identity = identity.squeeze()
                else:
                    identity = layer(identity)
            #identity = self.downsample.forward_for_jacobian(x)

        out += identity
        out = self.relu(out)

        return out

    def set_bn_training(self, training):
        # Set batchnorm training to True or False, without touching anything else.
        # This is for functorch jacobian calculation.
        self.bn1.training = training
        self.bn2.training = training

        if self.downsample is not None:
            for layer in self.downsample:
                if isinstance(layer, nn.BatchNorm2d):
                    layer.training = training


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        activation='relu',
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.relu = nn.GELU()
        elif activation == 'tanh':
            self.relu = nn.Tanh()
        else:
            raise AssertionError()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def forward_for_jacobian(self, x):
        identity = x

        out = self.conv1(x)
        # BN must be handled for torchfunc jvp
        out = out.unsqueeze(0)
        out = self.bn1(out)
        out = out.squeeze()
        out = self.relu(out)

        out = self.conv2(out)
        out = out.unsqueeze(0)
        out = self.bn2(out)
        out = out.squeeze()
        out = self.relu(out)

        out = self.conv3(out)
        out = out.unsqueeze(0)
        out = self.bn3(out)
        out = out.squeeze()

        if self.downsample is not None:
            for layer in self.downsample:
                if isinstance(layer, nn.BatchNorm2d):
                    identity = identity.unsqueeze(0)
                    identity = layer(identity)
                    identity = identity.squeeze()
                else:
                    identity = layer(identity)

        out += identity
        out = self.relu(out)

        return out

    def set_bn_training(self, training):
        # Set batchnorm training to True or False, without touching anything else.
        # This is for functorch jacobian calculation.
        self.bn1.training = training
        self.bn2.training = training
        self.bn3.training = training

        if self.downsample is not None:
            for layer in self.downsample:
                if isinstance(layer, nn.BatchNorm2d):
                    layer.training = training


# python3 train.py --dataset cifar10 --model resnet18 --activation gelu --bs 128 --lr 0.1 
                    # --weight-decay 5e-4 --standardize --nesterov --test-fil --pooling avg
                    #  --seed 123 --split-layer 7 --bottleneck-dim 8 --train-lb 1.0 
                    # --jvp-parallelism 100 --jacloss-alpha 0.0 --save-model

# split layer [2,3,5,7,9,11,12,13]
                    
class ResNet(nn.Module):
    def __init__(
        self,
        block, # 输入
        layers, # 输入
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        out_channels=[64, 128, 256, 512],
        split_layer=-1,
        bottleneck_dim=-1, # 是否要使用压缩层宽的方式来进行
        activation='relu',
        pooling='max'
    ):
        super(ResNet, self).__init__()

        # normalization layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # ?conv layer?
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        first_out = self.inplanes
        # END

        # activations layer
        if activation == 'relu':
            self.act = nn.ReLU
        elif activation == 'gelu':
            self.act = nn.GELU
        elif activation == 'tanh':
            self.act = nn.Tanh
        else:
            raise AssertionError('Unknown activation')

        # 
        self.bn1 = norm_layer(self.inplanes)
        self.relu = self.act()

        # pooling layer
        if pooling == 'max':
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif pooling == 'avg':
            maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise AssertionError('Unknown pooling')
        
        # make layers
        layer1 = self._make_layer(block, out_channels[0], layers[0], activation=activation)
        layer2 = self._make_layer(
            block, out_channels[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0], activation=activation
        )
        layer3 = self._make_layer(
            block, out_channels[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1], activation=activation
        )
        layer4 = self._make_layer(
            block, out_channels[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2], activation=activation
        )
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fc = nn.Linear(out_channels[3] * block.expansion, num_classes)

        # Test: Adding manual bottleneck layer
        # split layer [2,3,5,7,9,11,12,13]
        self.split_layer = split_layer
        self.layers = [self.conv1, self.bn1, self.relu, maxpool] + \
        list(layer1) + list(layer2) + list(layer3) + list(layer4) + \
        [avgpool, fc]

        # self.layers = nn.ModuleList(self.layers)
        self.selected_layers = nn.ModuleList(self.layers[:split_layer + 1])


        print(f"Num unit layers: {len(layers)}") # Num layers: 14
        print('Split layer:', self.split_layer)

        if bottleneck_dim > 0:
            sl = self.layers[split_layer]
            print(f"Split after {sl}")
            if sl in self.layers[:3]:
                in_dim = first_out
            elif sl in self.layer1:
                in_dim = out_channels[0] * block.expansion
            elif sl in self.layer2:
                in_dim = out_channels[1] * block.expansion
            elif sl in self.layer3:
                in_dim = out_channels[2] * block.expansion
            elif sl in self.layer4:
                in_dim = out_channels[3] * block.expansion

            # keep w/h and reduce c significantly # 这是一种防御手段吧！
            self.compress = nn.Sequential(
                    nn.Conv2d(in_dim, bottleneck_dim, kernel_size=3, padding=1),
                    #nn.BatchNorm2d, # Don't put BN yet because it is tedious to make it work with JVP..unless we need to
                    #nn.ReLU(inplace=False))
                    self.act())

            self.decompress = nn.Sequential(
                    nn.Conv2d(bottleneck_dim, in_dim, kernel_size=3, padding=1),
                    #nn.BatchNorm2d, # Don't put BN yet because it is tedious to make it work with JVP..unless we need to
                    self.act())
        else:
            self.compress = None
            self.decompress = None

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def create_new_fc(self, num_classes_new=-1):
        # TODO: Hacky code to make both applications work.
        if num_classes_new > -1:
            self.fc = nn.Linear(self.fc.weight.shape[1], num_classes_new)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, activation='relu'):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                activation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    activation=activation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        for i, layer in enumerate(self.selected_layers):
            if i == 13: # 13 is the last layer FC layer
                x = x.reshape(x.size(0), -1)
            x = layer(x)
            # if i == self.split_layer:
            #     if self.compress is not None:
            #         x = self.compress(x)
            #     if self.compress is not None:
            #         x = self.decompress(x)
                

        # x = self.avgpool(x)
        # x = self.fc(x)

        # 改成，仅包含split layer的形式。
        return x

    def forward_until_emb(self, x):
        print(self.layers)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        for i, layer in enumerate(self.layer4):
            if i < len(self.layer4) - 1:
                print("Do entirely ", layer) 
                x = layer(x)
            else:
                print("Skip the last relu of", layer)
                identity = x

                out = layer.conv1(x)
                out = layer.bn1(out)
                out = layer.relu(out)

                out = layer.conv2(out)
                out = layer.bn2(out)

                if layer.downsample is not None:
                    identity = layer.downsample(x)

                out += identity
                # Try just skipping the last relu
                #out = layer.relu(out)
                x = out

        #x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        return x

    def forward_first(self, x, for_jacobian=True):
        # For parallel jacobian calculation using functorch, we need to handle BN specially, hence a new forward.
        # Forward till split_layer and add Gaussian noise
        for layer in self.layers[:self.split_layer + 1]:
            if for_jacobian:
                if isinstance(layer, BasicBlock) or isinstance(layer, Bottleneck):
                    x = layer.forward_for_jacobian(x)
                elif isinstance(layer, nn.BatchNorm2d):
                    x = x.unsqueeze(0)
                    x = layer(x)
                    x = x.squeeze()
                else:
                    x = layer(x)
            else:
                x = layer(x)
        if self.compress is not None:
            x = self.compress(x)
        return x

    def forward_second(self, x, sigma=None):
        if sigma is not None:
            if len(sigma.shape) > 0:
                noise = torch.stack([torch.normal(torch.zeros_like(x[j]), sigma[j]) for j in range(len(sigma))])
            else:
                noise = torch.normal(torch.zeros_like(x), sigma)
            x = x + noise
        if self.compress is not None:
            x = self.decompress(x)
        for layer in self.layers[self.split_layer + 1:]:
            x = layer(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    def freeze_up_to_split(self):
        for i, layer in enumerate(self.layers):
            for param in layer.parameters():
                param.requires_grad = False
            if i == self.split_layer:
                return

    def set_bn_training(self, training):
        # Set batchnorm training to True or False, without touching anything else.
        # This is for functorch jacobian calculation.
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BasicBlock) or isinstance(layer, Bottleneck):
                layer.set_bn_training(training)
            elif isinstance(layer, nn.BatchNorm2d):
                layer.training = training

def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     script_dir = os.path.dirname(__file__) # 包含当前脚本文件的路径。os.path.dirname 函数用于获取该路径的目录部分。
    #     state_dict = torch.load(
    #         script_dir + "/state_dicts/" + arch + ".pt", map_location=device
    #     )
    #     model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, device='cpu', **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], **kwargs
    )

def resnet34(pretrained=False, device='cpu', **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], **kwargs
    )

def resnet50(pretrained=False, device='cpu', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], **kwargs
    )

# decoder
# assert(encoder_model == "resnet18")
# # InversionNet worked the best.
# if dataset == 'cifar10':
#     if split_layer <= 2:
#         inet = InversionNet(in_c=64, upconv_channels=[(128, 'same'), (3, 'same')], last_activation=last_activation)
#     elif split_layer == 7:
#         inet = InversionNet(last_activation=last_activation)
#     elif split_layer == 9:
#         inet = InversionNet(in_c=256, upconv_channels=[(256, 'up'), (256, 'up'), (3, 'up')], last_activation=last_activation)

resnet_model_cfg = {
    2: [(128, 'same'), (3, 'same')], # <=2
    3: [(128, 'up'), (3, 'same')],
    5: [(128, 'up'), (3, 'same')],
    7: [(128, 'up'), (3, 'up')],
    9: [(256, 'up'), (256, 'up'), (3, 'up')],
    11: [(512, 'up'), (512, 'up'), (256, 'up'), (3, 'up')], # drj
    12: [], # adaptive avgpooling这个怎么恢复，我还真不会。
    13: [],
}

in_c_dict = {
    2: 64,
    3: 64,
    5: 64,
    7: 128,
    9: 256,
    11: 512,
    12: 512,
    13: 512,
}
# resnet_model_cfg = {
#     'resnet18': [
#         ("conv1", 2)
#         ("pooling", 3)
#         ("layer11", 5)
#         ("layer21", 7)
#         ("layer31", 9)
#         ("layer41", 11)
#         ("avgpool", 12)
#         ("fc", 13)
#     ]
# }


class InversionNet(nn.Module):
    def __init__(
        self,
        num_conv=3, # 总共几个 conv layer
        # upconv_channels=[(128, 'up'), (3, 'up')],
        # in_c=128,
        # last_activation='sigmoid'
        last_activation=None,
        split_layer = 7,
    ):
        super(InversionNet, self).__init__()
        
        # 根据split layer 取参数
        upconv_channels = resnet_model_cfg[split_layer]
        in_c = in_c_dict[split_layer]

        layers = []

        # conv layer
        for _ in range(num_conv): # 不改变feature map size 和 通道数
            layers.append(nn.Conv2d(in_c, in_c, 3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # convtrans layer 
        for i, (c, mode) in enumerate(upconv_channels):
            if mode == 'up':
                layers.append(nn.ConvTranspose2d(in_channels=in_c, out_channels=c, kernel_size=3, stride=2, padding=1, output_padding=1))
            elif mode == 'same':
                layers.append(nn.ConvTranspose2d(in_channels=in_c, out_channels=c, kernel_size=3, stride=1, padding=1, output_padding=0))
            if i != len(upconv_channels) - 1:
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            in_c = c

        # TODO: Try Tanh?
        # if last_activation == 'sigmoid':
            # layers.append(nn.Sigmoid())
        # elif last_activation == 'tanh':
            # layers.append(nn.Tanh())
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        for layer in self.layers:
            x = layer(x)
            #print(layer)
        return x
